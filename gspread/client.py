"""
gspread.client
~~~~~~~~~~~~~~

This module contains Client class responsible for managing spreadsheet files

"""

from datetime import datetime
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Tuple, Union

from google.auth.credentials import Credentials
from requests import Response, Session

from .exceptions import APIError, SpreadsheetNotFound
from .http_client import HTTPClient, HTTPClientType, ParamsType
from .listing import (
    UNKNOWN_CROSS_PAGE_DUPLICATE,
    UNKNOWN_DUPLICATE,
    UNKNOWN_INCOMPLETE_SEARCH,
    UNKNOWN_MISSING_IDENTITY,
    UNKNOWN_PARTIAL_LISTING,
    UNKNOWN_PERMISSION_CHANGE,
    UNKNOWN_REPEATED_PAGE_TOKEN,
    UNKNOWN_VISIBILITY_CHANGED,
    PageCursor,
    SpreadsheetFileList,
    SpreadsheetFileListing,
    VISIBILITY_UNKNOWN,
    visibility_basis_for_file,
)
from .spreadsheet import Spreadsheet
from .urls import DRIVE_FILES_API_V3_COMMENTS_URL, DRIVE_FILES_API_V3_URL
from .utils import ExportFormat, MimeType, extract_id_from_url, finditem


class Client:
    """An instance of this class Manages Spreadsheet files

    It is used to:
        - open/create/list/delete spreadsheets
        - create/delete/list spreadsheet permission
        - etc

    It is the gspread entry point.
    It will handle creating necessary :class:`~gspread.models.Spreadsheet` instances.
    """

    def __init__(
        self,
        auth: Credentials,
        session: Optional[Session] = None,
        http_client: HTTPClientType = HTTPClient,
    ) -> None:
        self.http_client = http_client(auth, session)

    @property
    def expiry(self) -> Optional[datetime]:
        """Returns the expiry date of the curenlty loaded credentials

        :returns: (optional) datetime the expiry date time object.

        .. note::

           It only applies to gspread client created using oauth
        """
        return self.http_client.auth.expiry

    def set_timeout(
        self, timeout: Optional[Union[float, Tuple[float, float]]] = None
    ) -> None:
        """How long to wait for the server to send
        data before giving up, as a float, or a ``(connect timeout,
        read timeout)`` tuple.

        Use value ``None`` to restore default timeout

        Value for ``timeout`` is in seconds (s).
        """
        self.http_client.set_timeout(timeout)

    def get_file_drive_metadata(self, id: str) -> Any:
        """Get the metadata from the Drive API for a specific file
        This method is mainly here to retrieve the create/update time
        of a file (these metadata are only accessible from the Drive API).
        """
        return self.http_client.get_file_drive_metadata(id)

    def list_spreadsheet_files(
        self,
        title: Optional[str] = None,
        folder_id: Optional[str] = None,
        *,
        page_token: Optional[str] = None,
        include_visibility: bool = False,
        page_size: int = 1000,
    ) -> SpreadsheetFileList:
        """List all the spreadsheet files

        Will list all spreadsheet files owned by/shared with this user account.

        Files are deduplicated by their stable Drive file ID, never by title.
        Two different files with the same title are both returned.  The
        returned list also exposes listing metadata through its
        :attr:`~gspread.listing.SpreadsheetFileList.listing` attribute:
        the query scope, every page cursor, the basis on which each file is
        visible, and whether the enumeration is complete.

        :param str title: Filter only spreadsheet files with this title
        :param str folder_id: Only look for spreadsheet files in this folder
            The parameter ``folder_id`` can be obtained from the URL when looking at
            a folder in a web browser as follow:
            ``https://drive.google.com/drive/u/0/folders/<folder_id>``
        :param str page_token: (optional) Start from this Drive page cursor
            instead of the first page.  A listing that starts from a cursor is
            partial and is never used to confirm deletions.
        :param bool include_visibility: (optional) Also request the Drive
            ``ownedByMe``, ``shared`` and ``driveId`` fields.  This changes the
            request made to Drive, but gives :attr:`visibility_basis` a
            definite value instead of ``unknown``.
        :param int page_size: (optional) Number of files requested per page.

        :returns: a list of dicts containing the keys id, name, createdTime and modifiedTime.
        :rtype: :class:`~gspread.listing.SpreadsheetFileList`
        """
        listing = self._list_spreadsheet_files_detailed(
            title=title,
            folder_id=folder_id,
            page_token=page_token,
            include_visibility=include_visibility,
            page_size=page_size,
        )
        return SpreadsheetFileList(listing.files, listing)

    def list_spreadsheet_files_detailed(
        self,
        title: Optional[str] = None,
        folder_id: Optional[str] = None,
        *,
        page_token: Optional[str] = None,
        include_visibility: bool = False,
        page_size: int = 1000,
    ) -> SpreadsheetFileListing:
        """List spreadsheet files and return the full listing metadata.

        This is the same operation as :meth:`list_spreadsheet_files`, but it
        returns a :class:`~gspread.listing.SpreadsheetFileListing` instead of
        a plain list.  In addition to the files it keeps the exact query
        scope, the page cursor used for each request, the visibility basis for
        each file, and the reasons why the listing is not a complete
        enumeration (if any).
        """
        return self._list_spreadsheet_files_detailed(
            title=title,
            folder_id=folder_id,
            page_token=page_token,
            include_visibility=include_visibility,
            page_size=page_size,
        )

    def _spreadsheet_files_query(
        self, title: Optional[str], folder_id: Optional[str]
    ) -> str:
        query = f'mimeType="{MimeType.google_sheets}"'
        if title:
            escaped_title = title.replace("\\", "\\\\").replace('"', '\\"')
            query += f' and name = "{escaped_title}"'
        if folder_id:
            escaped_folder = folder_id.replace("\\", "\\\\").replace('"', '\\"')
            query += f' and parents in "{escaped_folder}"'
        return query

    def _spreadsheet_files_fields(self, include_visibility: bool) -> str:
        fields = "id,name,createdTime,modifiedTime"
        if include_visibility:
            fields += ",ownedByMe,shared,driveId"
            return "kind,nextPageToken,incompleteSearch,files({})".format(fields)
        # Keep the historical request unchanged for existing callers and
        # recorded fixtures.  incompleteSearch is still honoured when Drive
        # includes it in a response.
        return "kind,nextPageToken,files({})".format(fields)

    def _list_spreadsheet_files_detailed(
        self,
        title: Optional[str] = None,
        folder_id: Optional[str] = None,
        *,
        page_token: Optional[str] = None,
        include_visibility: bool = False,
        page_size: int = 1000,
    ) -> SpreadsheetFileListing:
        query = self._spreadsheet_files_query(title, folder_id)
        params: ParamsType = {
            "q": query,
            "pageSize": page_size,
            "supportsAllDrives": True,
            "includeItemsFromAllDrives": True,
            "fields": self._spreadsheet_files_fields(include_visibility),
        }

        listing = SpreadsheetFileListing(
            query_scope={
                "title": title,
                "folder_id": folder_id,
                "mime_type": str(MimeType.google_sheets),
                "q": query,
                "params": dict(params),
                "include_visibility": include_visibility,
            },
            requested_page_token=page_token,
            finished=False,
        )

        if page_token is not None:
            listing.partial = True
            if UNKNOWN_PARTIAL_LISTING not in listing.unknown_reasons:
                listing.unknown_reasons.append(UNKNOWN_PARTIAL_LISTING)

        seen_ids: Dict[str, int] = {}
        page_index = 0
        request_token = page_token
        requested_tokens = set()

        while True:
            if request_token is not None:
                if request_token in requested_tokens:
                    if UNKNOWN_REPEATED_PAGE_TOKEN not in listing.unknown_reasons:
                        listing.unknown_reasons.append(UNKNOWN_REPEATED_PAGE_TOKEN)
                    listing.finished = False
                    break
                requested_tokens.add(request_token)
                params["pageToken"] = request_token
            else:
                params.pop("pageToken", None)

            try:
                response = self.http_client.request(
                    "get", DRIVE_FILES_API_V3_URL, params=params
                )
            except APIError as ex:
                if ex.response.status_code in (
                    HTTPStatus.FORBIDDEN,
                    HTTPStatus.NOT_FOUND,
                    HTTPStatus.GONE,
                ):
                    listing.last_response = ex.response
                    listing.error = ex
                    if UNKNOWN_PERMISSION_CHANGE not in listing.unknown_reasons:
                        listing.unknown_reasons.append(UNKNOWN_PERMISSION_CHANGE)
                    listing.finished = False
                    break
                raise

            response_json = response.json()
            page_files = list(response_json.get("files", []))
            page_ids: List[str] = []

            for file in page_files:
                file_id = file.get("id")
                basis = visibility_basis_for_file(file)

                if not file_id:
                    listing.unidentified_files.append(file)
                    listing.files.append(file)
                    if UNKNOWN_MISSING_IDENTITY not in listing.unknown_reasons:
                        listing.unknown_reasons.append(UNKNOWN_MISSING_IDENTITY)
                    continue

                page_ids.append(file_id)

                if file_id in seen_ids:
                    duplicate_reason = (
                        UNKNOWN_DUPLICATE
                        if seen_ids[file_id] == page_index
                        else UNKNOWN_CROSS_PAGE_DUPLICATE
                    )
                    if duplicate_reason not in listing.unknown_reasons:
                        listing.unknown_reasons.append(duplicate_reason)
                    if file_id not in listing.duplicate_ids:
                        listing.duplicate_ids.append(file_id)

                    previous_basis = listing.visibility.get(file_id)
                    if (
                        previous_basis is not None
                        and basis != VISIBILITY_UNKNOWN
                        and previous_basis != VISIBILITY_UNKNOWN
                        and basis != previous_basis
                        and UNKNOWN_VISIBILITY_CHANGED
                        not in listing.unknown_reasons
                    ):
                        listing.unknown_reasons.append(UNKNOWN_VISIBILITY_CHANGED)

                    retained = listing.by_id.get(file_id)
                    if retained is not None:
                        for key, value in file.items():
                            retained.setdefault(key, value)
                    continue

                seen_ids[file_id] = page_index
                listing.files.append(file)
                listing.page_by_id[file_id] = request_token
                listing.visibility[file_id] = basis

            next_token = response_json.get("nextPageToken", None)
            incomplete_search = bool(
                response_json.get("incompleteSearch", False)
            )

            listing.page_cursors.append(
                PageCursor(
                    request_token=request_token,
                    next_token=next_token,
                    file_ids=tuple(page_ids),
                    incomplete_search=incomplete_search,
                    page=page_index,
                )
            )
            listing.last_response = response

            if (
                incomplete_search
                and UNKNOWN_INCOMPLETE_SEARCH not in listing.unknown_reasons
            ):
                listing.unknown_reasons.append(UNKNOWN_INCOMPLETE_SEARCH)

            page_index += 1

            if next_token is None:
                listing.finished = True
                break

            request_token = next_token

        return listing

    def _list_spreadsheet_files(
        self, title: Optional[str] = None, folder_id: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], Response]:
        listing = self._list_spreadsheet_files_detailed(
            title=title, folder_id=folder_id
        )
        files = SpreadsheetFileList(listing.files, listing)
        return files, listing.last_response

    def open(self, title: str, folder_id: Optional[str] = None) -> Spreadsheet:
        """Opens a spreadsheet.

        :param str title: A title of a spreadsheet.
        :param str folder_id: (optional) If specified can be used to filter
            spreadsheets by parent folder ID.
        :returns: a :class:`~gspread.models.Spreadsheet` instance.

        If there's more than one spreadsheet with same title the first one
        will be opened.

        :raises gspread.SpreadsheetNotFound: if no spreadsheet with
                                             specified `title` is found.

        >>> gc.open('My fancy spreadsheet')
        """
        spreadsheet_files, response = self._list_spreadsheet_files(title, folder_id)
        try:
            properties = finditem(
                lambda x: x["name"] == title,
                spreadsheet_files,
            )
        except StopIteration as ex:
            raise SpreadsheetNotFound(response) from ex

        # Drive uses different terminology
        properties["title"] = properties["name"]

        return Spreadsheet(self.http_client, properties)

    def open_by_key(self, key: str) -> Spreadsheet:
        """Opens a spreadsheet specified by `key` (a.k.a Spreadsheet ID).

        :param str key: A key of a spreadsheet as it appears in a URL in a browser.
        :returns: a :class:`~gspread.models.Spreadsheet` instance.

        >>> gc.open_by_key('0BmgG6nO_6dprdS1MN3d3MkdPa142WFRrdnRRUWl1UFE')
        """
        try:
            spreadsheet = Spreadsheet(self.http_client, {"id": key})
        except APIError as ex:
            if ex.response.status_code == HTTPStatus.NOT_FOUND:
                raise SpreadsheetNotFound(ex.response) from ex
            if ex.response.status_code == HTTPStatus.FORBIDDEN:
                raise PermissionError from ex
            raise ex
        return spreadsheet

    def open_by_url(self, url: str) -> Spreadsheet:
        """Opens a spreadsheet specified by `url`.

        :param str url: URL of a spreadsheet as it appears in a browser.

        :returns: a :class:`~gspread.models.Spreadsheet` instance.

        :raises gspread.SpreadsheetNotFound: if no spreadsheet with
                                             specified `url` is found.

        >>> gc.open_by_url('https://docs.google.com/spreadsheet/ccc?key=0Bm...FE&hl')
        """
        return self.open_by_key(extract_id_from_url(url))

    def openall(self, title: Optional[str] = None) -> List[Spreadsheet]:
        """Opens all available spreadsheets.

        :param str title: (optional) If specified can be used to filter
            spreadsheets by title.

        :returns: a list of :class:`~gspread.models.Spreadsheet` instances.
        """
        spreadsheet_files = self.list_spreadsheet_files(title)

        if title:
            spreadsheet_files = [
                spread for spread in spreadsheet_files if title == spread["name"]
            ]

        return [
            Spreadsheet(self.http_client, dict(title=x["name"], **x))
            for x in spreadsheet_files
        ]

    def create(self, title: str, folder_id: Optional[str] = None) -> Spreadsheet:
        """Creates a new spreadsheet.

        :param str title: A title of a new spreadsheet.

        :param str folder_id: Id of the folder where we want to save
            the spreadsheet.

        :returns: a :class:`~gspread.models.Spreadsheet` instance.

        """
        payload: Dict[str, Any] = {
            "name": title,
            "mimeType": MimeType.google_sheets,
        }

        params: ParamsType = {
            "supportsAllDrives": True,
        }

        if folder_id is not None:
            payload["parents"] = [folder_id]

        r = self.http_client.request(
            "post", DRIVE_FILES_API_V3_URL, json=payload, params=params
        )
        spreadsheet_id = r.json()["id"]
        return self.open_by_key(spreadsheet_id)

    def export(self, file_id: str, format: str = ExportFormat.PDF) -> bytes:
        """Export the spreadsheet in the given format.

        :param str file_id: The key of the spreadsheet to export

        :param str format: The format of the resulting file.
            Possible values are:

                * ``ExportFormat.PDF``
                * ``ExportFormat.EXCEL``
                * ``ExportFormat.CSV``
                * ``ExportFormat.OPEN_OFFICE_SHEET``
                * ``ExportFormat.TSV``
                * ``ExportFormat.ZIPPED_HTML``

            See `ExportFormat`_ in the Drive API.

        :type format: :class:`~gspread.utils.ExportFormat`

        :returns bytes: The content of the exported file.

        .. _ExportFormat: https://developers.google.com/drive/api/guides/ref-export-formats
        """

        return self.http_client.export(file_id=file_id, format=format)

    def copy(
        self,
        file_id: str,
        title: Optional[str] = None,
        copy_permissions: bool = False,
        folder_id: Optional[str] = None,
        copy_comments: bool = True,
    ) -> Spreadsheet:
        """Copies a spreadsheet.

        :param str file_id: A key of a spreadsheet to copy.
        :param str title: (optional) A title for the new spreadsheet.

        :param bool copy_permissions: (optional) If True, copy permissions from
            the original spreadsheet to the new spreadsheet.

        :param str folder_id: Id of the folder where we want to save
            the spreadsheet.

        :param bool copy_comments: (optional) If True, copy the comments from
            the original spreadsheet to the new spreadsheet.

        :returns: a :class:`~gspread.models.Spreadsheet` instance.

        .. versionadded:: 3.1.0

        .. note::

           If you're using custom credentials without the Drive scope, you need to add
           ``https://www.googleapis.com/auth/drive`` to your OAuth scope in order to use
           this method.

           Example::

              scope = [
                  'https://www.googleapis.com/auth/spreadsheets',
                  'https://www.googleapis.com/auth/drive'
              ]

           Otherwise, you will get an ``Insufficient Permission`` error
           when you try to copy a spreadsheet.

        """
        url = "{}/{}/copy".format(DRIVE_FILES_API_V3_URL, file_id)

        payload: Dict[str, Any] = {
            "name": title,
            "mimeType": MimeType.google_sheets,
        }

        if folder_id is not None:
            payload["parents"] = [folder_id]

        params: ParamsType = {"supportsAllDrives": True}
        r = self.http_client.request("post", url, json=payload, params=params)
        spreadsheet_id = r.json()["id"]

        new_spreadsheet = self.open_by_key(spreadsheet_id)

        if copy_permissions is True:
            original = self.open_by_key(file_id)

            permissions = original.list_permissions()
            for p in permissions:
                if p.get("deleted") or p.get("role") == "owner":
                    continue

                # In case of domain type the domain extract the domain
                # In case of user/group extract the emailAddress
                # Otherwise use None for type 'Anyone'

                email_or_domain = ""
                if str(p["type"]) == "domain":
                    email_or_domain = str(p["domain"])
                elif str(p["type"]) in ("user", "group"):
                    email_or_domain = str(p["emailAddress"])

                new_spreadsheet.share(
                    email_address=email_or_domain,
                    perm_type=str(p["type"]),
                    role=str(p["role"]),
                    notify=False,
                )

        if copy_comments is True:
            source_url = DRIVE_FILES_API_V3_COMMENTS_URL % (file_id)
            page_token = ""
            comments = []
            params = {
                "fields": "comments/content,comments/anchor,nextPageToken",
                "includeDeleted": False,
                "pageSize": 100,  # API limit to maximum 100
            }

            while page_token is not None:
                params["pageToken"] = page_token
                res = self.http_client.request("get", source_url, params=params).json()

                comments.extend(res["comments"])
                page_token = res.get("nextPageToken", None)

            destination_url = DRIVE_FILES_API_V3_COMMENTS_URL % (new_spreadsheet.id)
            # requesting some fields in the response is mandatory from the API.
            # choose 'id' randomly out of all the fields, but no need to use it for now.
            params = {"fields": "id"}
            for comment in comments:
                self.http_client.request(
                    "post", destination_url, json=comment, params=params
                )

        return new_spreadsheet

    def del_spreadsheet(self, file_id: str) -> None:
        """Deletes a spreadsheet.

        :param str file_id: a spreadsheet ID (a.k.a file ID).
        """
        url = "{}/{}".format(DRIVE_FILES_API_V3_URL, file_id)

        params: ParamsType = {"supportsAllDrives": True}
        self.http_client.request("delete", url, params=params)

    def import_csv(self, file_id: str, data: Union[str, bytes]) -> Any:
        """Imports data into the first page of the spreadsheet.

        :param str file_id:
        :param str data: A CSV string of data.

        Example:

        .. code::

            # Read CSV file contents
            content = open('file_to_import.csv', 'r').read()

            gc.import_csv(spreadsheet.id, content)

        .. note::

           This method removes all other worksheets and then entirely
           replaces the contents of the first worksheet.

        """
        return self.http_client.import_csv(file_id, data)

    def list_permissions(self, file_id: str) -> List[Dict[str, Union[str, bool]]]:
        """Retrieve a list of permissions for a file.

        :param str file_id: a spreadsheet ID (aka file ID).
        """
        return self.http_client.list_permissions(file_id)

    def insert_permission(
        self,
        file_id: str,
        value: Optional[str] = None,
        perm_type: Optional[str] = None,
        role: Optional[str] = None,
        notify: bool = True,
        email_message: Optional[str] = None,
        with_link: bool = False,
    ) -> Response:
        """Creates a new permission for a file.

        :param str file_id: a spreadsheet ID (aka file ID).
        :param value: user or group e-mail address, domain name
            or None for 'anyone' type.
        :type value: str, None
        :param str perm_type: (optional) The account type.
            Allowed values are: ``user``, ``group``, ``domain``, ``anyone``
        :param str role: (optional) The primary role for this user.
            Allowed values are: ``owner``, ``writer``, ``reader``
        :param bool notify: (optional) Whether to send an email to the target
            user/domain.
        :param str email_message: (optional) An email message to be sent
            if ``notify=True``.
        :param bool with_link: (optional) Whether the link is required for this
            permission to be active.

        :returns dict: the newly created permission

        Examples::

            # Give write permissions to otto@example.com

            gc.insert_permission(
                '0BmgG6nO_6dprnRRUWl1UFE',
                'otto@example.org',
                perm_type='user',
                role='writer'
            )

            # Make the spreadsheet publicly readable

            gc.insert_permission(
                '0BmgG6nO_6dprnRRUWl1UFE',
                None,
                perm_type='anyone',
                role='reader'
            )

        """
        return self.http_client.insert_permission(
            file_id, value, perm_type, role, notify, email_message, with_link
        )

    def remove_permission(self, file_id: str, permission_id: str) -> None:
        """Deletes a permission from a file.

        :param str file_id: a spreadsheet ID (aka file ID.)
        :param str permission_id: an ID for the permission.
        """
        self.http_client.remove_permission(file_id, permission_id)

<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <xsl:output method="xml" encoding="utf-8" indent="yes" />
  <xsl:template match="/">
    <xsl:for-each select="inp">
      <xsl:element name="out" />
    </xsl:for-each>
  </xsl:template>
</xsl:stylesheet>
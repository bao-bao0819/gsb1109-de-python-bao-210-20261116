.. code-block:: python

	#!/usr/bin/env python
	"""
	Assign isolated precursors to candidate molecules when the precursor charge
	state is not known in advance.
	
	The candidate catalogue states neutral masses and per-molecule mass windows.
	Every feasible charge/adduct interpretation is enumerated first, so a
	precursor that can be explained by two molecules under different charge
	states is reported as ambiguous instead of silently forcing one charge and
	declaring the other molecule undetected.
	
	usage:
	
	    ./assign_precursors.py
	"""
	
	from pprint import pprint
	
	import pymzml
	from pymzml.assignment import (
	    CandidateMolecule,
	    Precursor,
	    PROTON_MASS,
	    assign,
	)
	
	
	def main():
	    molecules = [
	        # neutral mass in Da, window in ppm of the theoretical m/z
	        CandidateMolecule("M_100", neutral_mass=100.0, tolerance=5.0),
	        CandidateMolecule("M_200", neutral_mass=200.0001, tolerance=5.0),
	        CandidateMolecule("M_300", neutral_mass=300.0, tolerance=0.01,
	                          tolerance_mode="da"),
	    ]
	
	    precursors = [
	        # uncertain charge: both z=1 and z=2 must be explored
	        Precursor(
	            precursor_id="scan-1",
	            isolation_event_id="event-1",
	            mz=101.0073,
	            charges=(1, 2),
	        ),
	        # clean, uniquely attributable precursor in its own event
	        Precursor(
	            precursor_id="scan-2",
	            isolation_event_id="event-2",
	            mz=300.0 + PROTON_MASS,
	            charges=(1,),
	        ),
	        # no feasible interpretation inside any declared window
	        Precursor(
	            precursor_id="scan-3",
	            isolation_event_id="event-3",
	            mz=999.9,
	            charges=(1, 2, 3),
	        ),
	    ]
	
	    print("=== single-molecule-per-isolation-event policy (default) ===")
	    result = assign(
	        precursors,
	        molecules,
	        default_tolerance=5.0,
	        allow_multiple_molecules_per_event=False,
	    )
	    for claim in result.unique:
	        print(
	            "UNIQUE  {0} <- {1} (charge/adduct ambiguous: {2})".format(
	                claim.molecule.molecule_id,
	                claim.precursor_ids,
	                claim.charge_adduct_ambiguous,
	            )
	        )
	    for group in result.ambiguous_groups:
	        print(
	            "AMBIGUOUS group {0}: molecules {1} share precursors {2}".format(
	                group.group_id,
	                group.molecule_ids,
	                group.precursor_ids,
	            )
	        )
	        print("  reasons: {0}".format(group.reasons))
	    for item in result.unassigned:
	        print(
	            "UNASSIGNED {0}: {1}".format(
	                item.precursor.precursor_id, item.reason
	            )
	        )
	
	    print()
	    print("=== every feasible interpretation with retained errors ===")
	    for interpretation in result.interpretations:
	        print(
	            "{precursor_id}: {molecule_id} z={charge} {adduct} "
	            "obs={observed_mz:.4f} theo={theoretical_mz:.4f} "
	            "err={mass_error_ppm:.2f} ppm".format(
	                **interpretation.as_dict()
	            )
	        )
	
	    print()
	    print("=== serialisable result payload (first level keys) ===")
	    pprint(sorted(result.as_dict().keys()))
	
	
	if __name__ == "__main__":
	    main()

import math
from numbers import Real
from qiskit import QuantumCircuit

PARAMETER_CONTRACT = {'constraints': [{'applies_when': 'always',
                  'id': 'positive_hamiltonian_scales',
                  'parameters': ['bare_coupling_g', 'lattice_spacing_a'],
                  'reason': 'The Hamiltonian contains g^(-2) and a^(-1).',
                  'requirement': 'bare_coupling_g > 0 and lattice_spacing_a > 0',
                  'severity': 'fatal'},
                 {'applies_when': 'the selected exact plaquette-dual gauge scheme is used',
                  'id': 'open_boundary_dual_reduction',
                  'parameters': [],
                  'reason': 'The selected one-qubit-per-plaquette reduction contains no '
                            'independent noncontractible holonomy degrees of freedom.',
                  'requirement': 'The spatial boundary condition must be open and the lattice '
                                 'must be connected and simply connected.',
                  'severity': 'fatal'},
                 {'applies_when': 'constructing either external state',
                  'id': 'physical_gauss_sector',
                  'parameters': [],
                  'reason': "Gauss's law zero denotes zero gauge charge, not eigenvalue zero.",
                  'requirement': 'The state must satisfy G_v=+1 at every lattice vertex; in '
                                 'the reduced representation each eliminated G_v must act as '
                                 'the identity.',
                  'severity': 'fatal'},
                 {'applies_when': 'constructing the glueball state',
                  'id': 'glueball_level_preservation',
                  'parameters': [],
                  'reason': 'The external-state level index is authoritative.',
                  'requirement': 'Use zero-based level_index=1 within the G_v=+1 physical '
                                 'sector and do not replace it by the sector ground state.',
                  'severity': 'fatal'},
                 {'applies_when': 'constructing the Hamiltonian',
                  'id': 'no_vacuum_energy_shift',
                  'parameters': ['bare_coupling_g', 'lattice_spacing_a'],
                  'reason': 'Vacuum subtraction belongs only to the requested matrix-element '
                            'combination.',
                  'requirement': 'Use the unshifted Hamiltonian with coefficients determined '
                                 'only by bare_coupling_g and lattice_spacing_a; do not add or '
                                 'subtract a target-state expectation value.',
                  'severity': 'fatal'}],
 'discretizations': [{'parameter_references': ['lattice_spacing_a'],
                      'scheme': 'finite square spatial lattice with continuous Hamiltonian '
                                'time',
                      'sector': 'spacetime',
                      'settings': {'horizontal_link_indices': {'x': [0, 2], 'y': [0, 3]},
                                   'independent_gauss_constraint_count': 15,
                                   'link_count': 24,
                                   'plaquette_count': 9,
                                   'plaquette_indices': {'x': [0, 2], 'y': [0, 2]},
                                   'spatial_boundary_condition': 'open',
                                   'spatial_shape': [4, 4],
                                   'time_realization': 'continuous',
                                   'vertex_count': 16,
                                   'vertex_indices': {'x': [0, 3], 'y': [0, 3]},
                                   'vertical_link_indices': {'x': [0, 3], 'y': [0, 2]}},
                      'truncation': {'reason': 'The theory is intrinsically a finite '
                                               'spatial-lattice Hamiltonian theory with '
                                               'continuous time.',
                                     'scheme': 'none',
                                     'settings': {}}},
                     {'parameter_references': ['bare_coupling_g', 'lattice_spacing_a'],
                      'scheme': 'exact Gauss-law reduction to open-boundary plaquette-dual '
                                'variables',
                      'sector': 'gauge',
                      'settings': {'dual_degrees_of_freedom': 'one Pauli qubit tau_p for each '
                                                              'plaquette defined by the '
                                                              'spacetime settings',
                                   'effective_qubit_count': 9,
                                   'electric_mapping': 'X_l maps to the product of tau_p^x '
                                                       'over plaquettes incident on link l',
                                   'explicit_embedding_complex128_bytes': 137438953472,
                                   'full_dense_complex128_bytes': 4503599627370496,
                                   'gauge_fixing': 'none',
                                   'gauge_group': 'Z2',
                                   'gauss_sector': 'G_v=+1 at every lattice vertex',
                                   'maximum_reduced_term_support': 2,
                                   'original_kinematic_dimension': 16777216,
                                   'original_link_local_dimension': 2,
                                   'physical_dimension': 512,
                                   'plaquette_mapping': 'B_p maps to tau_p^z',
                                   'reduced_dense_complex128_bytes': 4194304},
                      'truncation': {'reason': 'Z2 links are intrinsically finite-dimensional, '
                                               'and the plaquette-dual reduction is exactly '
                                               'unitarily equivalent to the full charge-free '
                                               'physical sector under OBC.',
                                     'scheme': 'none',
                                     'settings': {'finite_group_representation': 'complete '
                                                                                 'two-dimensional '
                                                                                 'Z2 link '
                                                                                 'representation',
                                                  'reduction_exact': True}}}],
 'external_states': {'definitions': {'glueball': 'Quantum-number-sector eigenstate; '
                                                 'level_index=1; quantum_numbers={"Gauss\'s '
                                                 'law": 0}',
                                     'vacuum': 'Ground state / vacuum'},
                     'equivalent_groups': [],
                     'labels': ['glueball', 'vacuum']},
 'model': {'gauge_structure': 'local Z2',
           'name': '2+1 dimensional pure Z2 gauge theory in Kogut-Susskind formalism',
           'spacetime_dimension': '2+1'},
 'parameters': [{'domain': {'minimum_exclusive': 0},
                 'name': 'bare_coupling_g',
                 'required_for_run': True,
                 'role': 'Bare coupling in the coefficients g^2 and g^(-2).',
                 'selected_value': 1,
                 'symbol': 'g',
                 'type': 'real',
                 'unit': None},
                {'domain': {'minimum_exclusive': 0},
                 'name': 'lattice_spacing_a',
                 'required_for_run': True,
                 'role': 'Spatial lattice spacing and inverse overall Hamiltonian energy '
                         'scale; temporal spacing is the same if a downstream time '
                         'discretization is introduced.',
                 'selected_value': 1,
                 'symbol': 'a',
                 'type': 'real',
                 'unit': 'chosen natural length unit'},
                {'domain': {'source': 'immutable RunSpec value'},
                 'immutable': True,
                 'name': 'bare copuling: g',
                 'required_for_run': True,
                 'role': 'Immutable user input; downstream stages must reference this path.',
                 'selected_value': 'g = 1',
                 'source_ref': '/user_input/parameters/bare copuling: g',
                 'symbol': 'bare copuling: g',
                 'type': 'string',
                 'unit': None},
                {'domain': {'source': 'immutable RunSpec value'},
                 'immutable': True,
                 'name': 'number of lattice site [L,L]:',
                 'required_for_run': True,
                 'role': 'Immutable user input; downstream stages must reference this path.',
                 'selected_value': [4, 4],
                 'source_ref': '/user_input/parameters/number of lattice site [L,L]:',
                 'symbol': 'number of lattice site [L,L]:',
                 'type': 'list',
                 'unit': None}],
 'schema_version': '1.1',
 'validation_obligations': [{'description': 'Confirm that both named external states lie in '
                                            'the zero-charge sector G_v=+1 at every lattice '
                                            'vertex, or equivalently in the exact reduced '
                                            'physical Hilbert space.',
                             'id': 'validate_physical_sector',
                             'severity': 'fatal',
                             'stage': 'pre_simulation'},
                            {'description': 'Confirm that the reduced plaquette and electric '
                                            'operators reproduce the physical-sector algebra '
                                            'of B_p and X_l, including all '
                                            'link-to-incident-plaquette relations at the open '
                                            'boundary.',
                             'id': 'validate_dual_operator_algebra',
                             'severity': 'fatal',
                             'stage': 'post_construction'},
                            {'description': 'Confirm that glueball is the zero-based level-1 '
                                            'Hamiltonian eigenstate in the physical Gauss '
                                            'sector, without imposing an unspecified D4 or '
                                            'other discrete-symmetry eigenvalue.',
                             'id': 'validate_glueball_level',
                             'severity': 'fatal',
                             'stage': 'post_construction'},
                            {'description': 'Confirm that no vacuum-energy subtraction or '
                                            'state-dependent coefficient has been inserted '
                                            'into the constructed Hamiltonian.',
                             'id': 'validate_unshifted_hamiltonian',
                             'severity': 'fatal',
                             'stage': 'post_construction'}]}

OBSERVABLE_SPECIFICATION = {'coordinate_convention': {'mode': 'lattice_indices',
                           'spacing_parameter': 'lattice_spacing_a',
                           'temporal_spacing': 'same_as_spatial',
                           'units': 'natural_units_c_1'},
 'expression': 'Hamiltonian: H',
 'insertions': [{'id': 'Hamiltonian', 'operator': 'H', 'position': 'global'}],
 'matrix_elements': [{'bra': 'glueball',
                      'coefficient': 1.0,
                      'id': 'glueball energy',
                      'ket': 'glueball'},
                     {'bra': 'vacuum',
                      'coefficient': -1.0,
                      'id': 'vacuum energy',
                      'ket': 'vacuum'}],
 'name': 'user_observable',
 'observable_schema_version': '1.0',
 'ordering': 'as_written',
 'requested_components': ['real', 'imag'],
 'sampling': {'mode': 'explicit_points', 'points': [{'sample_id': 'default', 'values': {}}]}}

_OPERATOR_DATA = (
    ("e_x_q0_m2", "IIIIIIIIX", "electric", 2),
    ("e_x_q5_q0", "IIIIIXIIX", "electric", 1),
    ("e_x_q6_q3", "IIXIIXIII", "electric", 1),
    ("e_x_q6_m2", "IIXIIIIII", "electric", 2),
    ("e_x_q1", "IIIIIIIXI", "electric", 1),
    ("e_x_q4_q1", "IIIIXIIXI", "electric", 1),
    ("e_x_q7_q4", "IXIIXIIII", "electric", 1),
    ("e_x_q7", "IXIIIIIII", "electric", 1),
    ("e_x_q2_m2", "IIIIIIXII", "electric", 2),
    ("e_x_q5_q2", "IIIXIIXII", "electric", 1),
    ("e_x_q8_q5", "XIIXIIIII", "electric", 1),
    ("e_x_q8_m2", "XIIIIIIII", "electric", 2),
    ("e_x_q3", "IIIIIXIII", "electric", 1),
    ("e_x_q1_q0", "IIIIIIIXX", "electric", 1),
    ("e_x_q4_q3", "IIIIXXIII", "electric", 1),
    ("e_x_q7_q6", "IXXIIIIII", "electric", 1),
    ("e_x_q2_q1", "IIIIIIXXI", "electric", 1),
    ("e_x_q5_q4", "IIIXXIIII", "electric", 1),
    ("e_x_q8_q7", "XXIIIIIII", "electric", 1),
    ("e_x_q5", "IIIXIIIII", "electric", 1),
    ("b_z_q0", "IIIIIIIIZ", "magnetic", 1),
    ("b_z_q1", "IIIIIIIZI", "magnetic", 1),
    ("b_z_q2", "IIIIIIZII", "magnetic", 1),
    ("b_z_q3", "IIIIIZIII", "magnetic", 1),
    ("b_z_q4", "IIIIZIIII", "magnetic", 1),
    ("b_z_q5", "IIIZIIIII", "magnetic", 1),
    ("b_z_q6", "IIZIIIIII", "magnetic", 1),
    ("b_z_q7", "IZIIIIIII", "magnetic", 1),
    ("b_z_q8", "ZIIIIIIII", "magnetic", 1),
)

_RECIPE_IDS = tuple(
    f"{state}_{operator_id}_real"
    for state in ("glueball", "vacuum")
    for operator_id, _, _, _ in _OPERATOR_DATA
)

OBSERVABLE_CONTRACT = {'adapter_interface': {'builder': 'build_observable_measurements',
                       'call_signature': 'build_observable_measurements(state_circuits, '
                                         'trotter_steps, sample_values)',
                       'reconstructor': 'reconstruct_observable',
                       'reconstructor_input_rule': 'Receive a mapping containing one '
                                                   'expectation for every exact '
                                                   'circuit_recipe_id plus the same empty '
                                                   'sample_values object, then apply '
                                                   'reconstruction.terms and return {real: '
                                                   'weighted_sum, imag: 0}.',
                       'required_sample_values': {},
                       'required_trotter_steps': 0,
                       'return_rule': 'Return all 58 circuit recipes in '
                                      'measurement.circuit_recipe_ids in one call. Each '
                                      'circuit is the unrotated supplied preparation circuit '
                                      'with the listed readout_pauli; do not append '
                                      'Pauli-basis rotations.'},
 'contract_type': 'observable_measurement',
 'literature_sources': [{'applicability': 'direct: the requested diagonal Hamiltonian matrix '
                                          'elements are weighted sums of Pauli expectations',
                         'locator': 'Equation (Hq) and the immediately preceding paragraph '
                                    'beginning "The update of the angles..." describing energy '
                                    'estimation from a weighted Pauli sum',
                         'source_path': 'arXiv-1704.05018v2.tar/HardwareEffEigensolver.tex'}],
 'measurement': {'ancilla_qubits': 0,
                 'circuit_recipe_ids': ['glueball_e_x_q0_m2_real',
                                        'glueball_e_x_q5_q0_real',
                                        'glueball_e_x_q6_q3_real',
                                        'glueball_e_x_q6_m2_real',
                                        'glueball_e_x_q1_real',
                                        'glueball_e_x_q4_q1_real',
                                        'glueball_e_x_q7_q4_real',
                                        'glueball_e_x_q7_real',
                                        'glueball_e_x_q2_m2_real',
                                        'glueball_e_x_q5_q2_real',
                                        'glueball_e_x_q8_q5_real',
                                        'glueball_e_x_q8_m2_real',
                                        'glueball_e_x_q3_real',
                                        'glueball_e_x_q1_q0_real',
                                        'glueball_e_x_q4_q3_real',
                                        'glueball_e_x_q7_q6_real',
                                        'glueball_e_x_q2_q1_real',
                                        'glueball_e_x_q5_q4_real',
                                        'glueball_e_x_q8_q7_real',
                                        'glueball_e_x_q5_real',
                                        'glueball_b_z_q0_real',
                                        'glueball_b_z_q1_real',
                                        'glueball_b_z_q2_real',
                                        'glueball_b_z_q3_real',
                                        'glueball_b_z_q4_real',
                                        'glueball_b_z_q5_real',
                                        'glueball_b_z_q6_real',
                                        'glueball_b_z_q7_real',
                                        'glueball_b_z_q8_real',
                                        'vacuum_e_x_q0_m2_real',
                                        'vacuum_e_x_q5_q0_real',
                                        'vacuum_e_x_q6_q3_real',
                                        'vacuum_e_x_q6_m2_real',
                                        'vacuum_e_x_q1_real',
                                        'vacuum_e_x_q4_q1_real',
                                        'vacuum_e_x_q7_q4_real',
                                        'vacuum_e_x_q7_real',
                                        'vacuum_e_x_q2_m2_real',
                                        'vacuum_e_x_q5_q2_real',
                                        'vacuum_e_x_q8_q5_real',
                                        'vacuum_e_x_q8_m2_real',
                                        'vacuum_e_x_q3_real',
                                        'vacuum_e_x_q1_q0_real',
                                        'vacuum_e_x_q4_q3_real',
                                        'vacuum_e_x_q7_q6_real',
                                        'vacuum_e_x_q2_q1_real',
                                        'vacuum_e_x_q5_q4_real',
                                        'vacuum_e_x_q8_q7_real',
                                        'vacuum_e_x_q5_real',
                                        'vacuum_b_z_q0_real',
                                        'vacuum_b_z_q1_real',
                                        'vacuum_b_z_q2_real',
                                        'vacuum_b_z_q3_real',
                                        'vacuum_b_z_q4_real',
                                        'vacuum_b_z_q5_real',
                                        'vacuum_b_z_q6_real',
                                        'vacuum_b_z_q7_real',
                                        'vacuum_b_z_q8_real'],
                 'circuit_recipes': [{'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_glueball'}],
                                      'readout_pauli': 'IIIIIIIIX',
                                      'recipe_id': 'glueball_e_x_q0_m2_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_glueball'}],
                                      'readout_pauli': 'IIIIIXIIX',
                                      'recipe_id': 'glueball_e_x_q5_q0_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_glueball'}],
                                      'readout_pauli': 'IIXIIXIII',
                                      'recipe_id': 'glueball_e_x_q6_q3_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_glueball'}],
                                      'readout_pauli': 'IIXIIIIII',
                                      'recipe_id': 'glueball_e_x_q6_m2_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_glueball'}],
                                      'readout_pauli': 'IIIIIIIXI',
                                      'recipe_id': 'glueball_e_x_q1_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_glueball'}],
                                      'readout_pauli': 'IIIIXIIXI',
                                      'recipe_id': 'glueball_e_x_q4_q1_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_glueball'}],
                                      'readout_pauli': 'IXIIXIIII',
                                      'recipe_id': 'glueball_e_x_q7_q4_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_glueball'}],
                                      'readout_pauli': 'IXIIIIIII',
                                      'recipe_id': 'glueball_e_x_q7_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_glueball'}],
                                      'readout_pauli': 'IIIIIIXII',
                                      'recipe_id': 'glueball_e_x_q2_m2_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_glueball'}],
                                      'readout_pauli': 'IIIXIIXII',
                                      'recipe_id': 'glueball_e_x_q5_q2_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_glueball'}],
                                      'readout_pauli': 'XIIXIIIII',
                                      'recipe_id': 'glueball_e_x_q8_q5_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_glueball'}],
                                      'readout_pauli': 'XIIIIIIII',
                                      'recipe_id': 'glueball_e_x_q8_m2_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_glueball'}],
                                      'readout_pauli': 'IIIIIXIII',
                                      'recipe_id': 'glueball_e_x_q3_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_glueball'}],
                                      'readout_pauli': 'IIIIIIIXX',
                                      'recipe_id': 'glueball_e_x_q1_q0_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_glueball'}],
                                      'readout_pauli': 'IIIIXXIII',
                                      'recipe_id': 'glueball_e_x_q4_q3_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_glueball'}],
                                      'readout_pauli': 'IXXIIIIII',
                                      'recipe_id': 'glueball_e_x_q7_q6_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_glueball'}],
                                      'readout_pauli': 'IIIIIIXXI',
                                      'recipe_id': 'glueball_e_x_q2_q1_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_glueball'}],
                                      'readout_pauli': 'IIIXXIIII',
                                      'recipe_id': 'glueball_e_x_q5_q4_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_glueball'}],
                                      'readout_pauli': 'XXIIIIIII',
                                      'recipe_id': 'glueball_e_x_q8_q7_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_glueball'}],
                                      'readout_pauli': 'IIIXIIIII',
                                      'recipe_id': 'glueball_e_x_q5_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_glueball'}],
                                      'readout_pauli': 'IIIIIIIIZ',
                                      'recipe_id': 'glueball_b_z_q0_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_glueball'}],
                                      'readout_pauli': 'IIIIIIIZI',
                                      'recipe_id': 'glueball_b_z_q1_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_glueball'}],
                                      'readout_pauli': 'IIIIIIZII',
                                      'recipe_id': 'glueball_b_z_q2_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_glueball'}],
                                      'readout_pauli': 'IIIIIZIII',
                                      'recipe_id': 'glueball_b_z_q3_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_glueball'}],
                                      'readout_pauli': 'IIIIZIIII',
                                      'recipe_id': 'glueball_b_z_q4_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_glueball'}],
                                      'readout_pauli': 'IIIZIIIII',
                                      'recipe_id': 'glueball_b_z_q5_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_glueball'}],
                                      'readout_pauli': 'IIZIIIIII',
                                      'recipe_id': 'glueball_b_z_q6_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_glueball'}],
                                      'readout_pauli': 'IZIIIIIII',
                                      'recipe_id': 'glueball_b_z_q7_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_glueball'}],
                                      'readout_pauli': 'ZIIIIIIII',
                                      'recipe_id': 'glueball_b_z_q8_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_vacuum'}],
                                      'readout_pauli': 'IIIIIIIIX',
                                      'recipe_id': 'vacuum_e_x_q0_m2_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_vacuum'}],
                                      'readout_pauli': 'IIIIIXIIX',
                                      'recipe_id': 'vacuum_e_x_q5_q0_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_vacuum'}],
                                      'readout_pauli': 'IIXIIXIII',
                                      'recipe_id': 'vacuum_e_x_q6_q3_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_vacuum'}],
                                      'readout_pauli': 'IIXIIIIII',
                                      'recipe_id': 'vacuum_e_x_q6_m2_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_vacuum'}],
                                      'readout_pauli': 'IIIIIIIXI',
                                      'recipe_id': 'vacuum_e_x_q1_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_vacuum'}],
                                      'readout_pauli': 'IIIIXIIXI',
                                      'recipe_id': 'vacuum_e_x_q4_q1_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_vacuum'}],
                                      'readout_pauli': 'IXIIXIIII',
                                      'recipe_id': 'vacuum_e_x_q7_q4_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_vacuum'}],
                                      'readout_pauli': 'IXIIIIIII',
                                      'recipe_id': 'vacuum_e_x_q7_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_vacuum'}],
                                      'readout_pauli': 'IIIIIIXII',
                                      'recipe_id': 'vacuum_e_x_q2_m2_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_vacuum'}],
                                      'readout_pauli': 'IIIXIIXII',
                                      'recipe_id': 'vacuum_e_x_q5_q2_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_vacuum'}],
                                      'readout_pauli': 'XIIXIIIII',
                                      'recipe_id': 'vacuum_e_x_q8_q5_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_vacuum'}],
                                      'readout_pauli': 'XIIIIIIII',
                                      'recipe_id': 'vacuum_e_x_q8_m2_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_vacuum'}],
                                      'readout_pauli': 'IIIIIXIII',
                                      'recipe_id': 'vacuum_e_x_q3_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_vacuum'}],
                                      'readout_pauli': 'IIIIIIIXX',
                                      'recipe_id': 'vacuum_e_x_q1_q0_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_vacuum'}],
                                      'readout_pauli': 'IIIIXXIII',
                                      'recipe_id': 'vacuum_e_x_q4_q3_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_vacuum'}],
                                      'readout_pauli': 'IXXIIIIII',
                                      'recipe_id': 'vacuum_e_x_q7_q6_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_vacuum'}],
                                      'readout_pauli': 'IIIIIIXXI',
                                      'recipe_id': 'vacuum_e_x_q2_q1_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_vacuum'}],
                                      'readout_pauli': 'IIIXXIIII',
                                      'recipe_id': 'vacuum_e_x_q5_q4_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_vacuum'}],
                                      'readout_pauli': 'XXIIIIIII',
                                      'recipe_id': 'vacuum_e_x_q8_q7_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_vacuum'}],
                                      'readout_pauli': 'IIIXIIIII',
                                      'recipe_id': 'vacuum_e_x_q5_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_vacuum'}],
                                      'readout_pauli': 'IIIIIIIIZ',
                                      'recipe_id': 'vacuum_b_z_q0_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_vacuum'}],
                                      'readout_pauli': 'IIIIIIIZI',
                                      'recipe_id': 'vacuum_b_z_q1_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_vacuum'}],
                                      'readout_pauli': 'IIIIIIZII',
                                      'recipe_id': 'vacuum_b_z_q2_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_vacuum'}],
                                      'readout_pauli': 'IIIIIZIII',
                                      'recipe_id': 'vacuum_b_z_q3_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_vacuum'}],
                                      'readout_pauli': 'IIIIZIIII',
                                      'recipe_id': 'vacuum_b_z_q4_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_vacuum'}],
                                      'readout_pauli': 'IIIZIIIII',
                                      'recipe_id': 'vacuum_b_z_q5_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_vacuum'}],
                                      'readout_pauli': 'IIZIIIIII',
                                      'recipe_id': 'vacuum_b_z_q6_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_vacuum'}],
                                      'readout_pauli': 'IZIIIIIII',
                                      'recipe_id': 'vacuum_b_z_q7_real'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'prepare_vacuum'}],
                                      'readout_pauli': 'ZIIIIIIII',
                                      'recipe_id': 'vacuum_b_z_q8_real'}],
                 'components': ['real', 'imag'],
                 'protocol_id': 'z2_4x4_obc_direct_combined_hamiltonian_difference_v1'},
 'operator_recipes': [{'hamiltonian_coefficient': '-2*bare_coupling_g^2/lattice_spacing_a',
                       'id': 'op_e_x_q0_m2',
                       'multiplicity': 2,
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'PauliString(IIIIIIIIX)',
                                       'operation': 'declare_readout_observable',
                                       'targets': ['q0']}]},
                      {'hamiltonian_coefficient': '-bare_coupling_g^2/lattice_spacing_a',
                       'id': 'op_e_x_q5_q0',
                       'multiplicity': 1,
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'PauliString(IIIIIXIIX)',
                                       'operation': 'declare_readout_observable',
                                       'targets': ['q5', 'q0']}]},
                      {'hamiltonian_coefficient': '-bare_coupling_g^2/lattice_spacing_a',
                       'id': 'op_e_x_q6_q3',
                       'multiplicity': 1,
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'PauliString(IIXIIXIII)',
                                       'operation': 'declare_readout_observable',
                                       'targets': ['q6', 'q3']}]},
                      {'hamiltonian_coefficient': '-2*bare_coupling_g^2/lattice_spacing_a',
                       'id': 'op_e_x_q6_m2',
                       'multiplicity': 2,
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'PauliString(IIXIIIIII)',
                                       'operation': 'declare_readout_observable',
                                       'targets': ['q6']}]},
                      {'hamiltonian_coefficient': '-bare_coupling_g^2/lattice_spacing_a',
                       'id': 'op_e_x_q1',
                       'multiplicity': 1,
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'PauliString(IIIIIIIXI)',
                                       'operation': 'declare_readout_observable',
                                       'targets': ['q1']}]},
                      {'hamiltonian_coefficient': '-bare_coupling_g^2/lattice_spacing_a',
                       'id': 'op_e_x_q4_q1',
                       'multiplicity': 1,
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'PauliString(IIIIXIIXI)',
                                       'operation': 'declare_readout_observable',
                                       'targets': ['q4', 'q1']}]},
                      {'hamiltonian_coefficient': '-bare_coupling_g^2/lattice_spacing_a',
                       'id': 'op_e_x_q7_q4',
                       'multiplicity': 1,
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'PauliString(IXIIXIIII)',
                                       'operation': 'declare_readout_observable',
                                       'targets': ['q7', 'q4']}]},
                      {'hamiltonian_coefficient': '-bare_coupling_g^2/lattice_spacing_a',
                       'id': 'op_e_x_q7',
                       'multiplicity': 1,
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'PauliString(IXIIIIIII)',
                                       'operation': 'declare_readout_observable',
                                       'targets': ['q7']}]},
                      {'hamiltonian_coefficient': '-2*bare_coupling_g^2/lattice_spacing_a',
                       'id': 'op_e_x_q2_m2',
                       'multiplicity': 2,
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'PauliString(IIIIIIXII)',
                                       'operation': 'declare_readout_observable',
                                       'targets': ['q2']}]},
                      {'hamiltonian_coefficient': '-bare_coupling_g^2/lattice_spacing_a',
                       'id': 'op_e_x_q5_q2',
                       'multiplicity': 1,
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'PauliString(IIIXIIXII)',
                                       'operation': 'declare_readout_observable',
                                       'targets': ['q5', 'q2']}]},
                      {'hamiltonian_coefficient': '-bare_coupling_g^2/lattice_spacing_a',
                       'id': 'op_e_x_q8_q5',
                       'multiplicity': 1,
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'PauliString(XIIXIIIII)',
                                       'operation': 'declare_readout_observable',
                                       'targets': ['q8', 'q5']}]},
                      {'hamiltonian_coefficient': '-2*bare_coupling_g^2/lattice_spacing_a',
                       'id': 'op_e_x_q8_m2',
                       'multiplicity': 2,
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'PauliString(XIIIIIIII)',
                                       'operation': 'declare_readout_observable',
                                       'targets': ['q8']}]},
                      {'hamiltonian_coefficient': '-bare_coupling_g^2/lattice_spacing_a',
                       'id': 'op_e_x_q3',
                       'multiplicity': 1,
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'PauliString(IIIIIXIII)',
                                       'operation': 'declare_readout_observable',
                                       'targets': ['q3']}]},
                      {'hamiltonian_coefficient': '-bare_coupling_g^2/lattice_spacing_a',
                       'id': 'op_e_x_q1_q0',
                       'multiplicity': 1,
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'PauliString(IIIIIIIXX)',
                                       'operation': 'declare_readout_observable',
                                       'targets': ['q1', 'q0']}]},
                      {'hamiltonian_coefficient': '-bare_coupling_g^2/lattice_spacing_a',
                       'id': 'op_e_x_q4_q3',
                       'multiplicity': 1,
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'PauliString(IIIIXXIII)',
                                       'operation': 'declare_readout_observable',
                                       'targets': ['q4', 'q3']}]},
                      {'hamiltonian_coefficient': '-bare_coupling_g^2/lattice_spacing_a',
                       'id': 'op_e_x_q7_q6',
                       'multiplicity': 1,
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'PauliString(IXXIIIIII)',
                                       'operation': 'declare_readout_observable',
                                       'targets': ['q7', 'q6']}]},
                      {'hamiltonian_coefficient': '-bare_coupling_g^2/lattice_spacing_a',
                       'id': 'op_e_x_q2_q1',
                       'multiplicity': 1,
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'PauliString(IIIIIIXXI)',
                                       'operation': 'declare_readout_observable',
                                       'targets': ['q2', 'q1']}]},
                      {'hamiltonian_coefficient': '-bare_coupling_g^2/lattice_spacing_a',
                       'id': 'op_e_x_q5_q4',
                       'multiplicity': 1,
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'PauliString(IIIXXIIII)',
                                       'operation': 'declare_readout_observable',
                                       'targets': ['q5', 'q4']}]},
                      {'hamiltonian_coefficient': '-bare_coupling_g^2/lattice_spacing_a',
                       'id': 'op_e_x_q8_q7',
                       'multiplicity': 1,
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'PauliString(XXIIIIIII)',
                                       'operation': 'declare_readout_observable',
                                       'targets': ['q8', 'q7']}]},
                      {'hamiltonian_coefficient': '-bare_coupling_g^2/lattice_spacing_a',
                       'id': 'op_e_x_q5',
                       'multiplicity': 1,
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'PauliString(IIIXIIIII)',
                                       'operation': 'declare_readout_observable',
                                       'targets': ['q5']}]},
                      {'hamiltonian_coefficient': '-bare_coupling_g^(-2)/lattice_spacing_a',
                       'id': 'op_b_z_q0',
                       'multiplicity': 1,
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'PauliString(IIIIIIIIZ)',
                                       'operation': 'declare_readout_observable',
                                       'targets': ['q0']}]},
                      {'hamiltonian_coefficient': '-bare_coupling_g^(-2)/lattice_spacing_a',
                       'id': 'op_b_z_q1',
                       'multiplicity': 1,
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'PauliString(IIIIIIIZI)',
                                       'operation': 'declare_readout_observable',
                                       'targets': ['q1']}]},
                      {'hamiltonian_coefficient': '-bare_coupling_g^(-2)/lattice_spacing_a',
                       'id': 'op_b_z_q2',
                       'multiplicity': 1,
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'PauliString(IIIIIIZII)',
                                       'operation': 'declare_readout_observable',
                                       'targets': ['q2']}]},
                      {'hamiltonian_coefficient': '-bare_coupling_g^(-2)/lattice_spacing_a',
                       'id': 'op_b_z_q3',
                       'multiplicity': 1,
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'PauliString(IIIIIZIII)',
                                       'operation': 'declare_readout_observable',
                                       'targets': ['q3']}]},
                      {'hamiltonian_coefficient': '-bare_coupling_g^(-2)/lattice_spacing_a',
                       'id': 'op_b_z_q4',
                       'multiplicity': 1,
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'PauliString(IIIIZIIII)',
                                       'operation': 'declare_readout_observable',
                                       'targets': ['q4']}]},
                      {'hamiltonian_coefficient': '-bare_coupling_g^(-2)/lattice_spacing_a',
                       'id': 'op_b_z_q5',
                       'multiplicity': 1,
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'PauliString(IIIZIIIII)',
                                       'operation': 'declare_readout_observable',
                                       'targets': ['q5']}]},
                      {'hamiltonian_coefficient': '-bare_coupling_g^(-2)/lattice_spacing_a',
                       'id': 'op_b_z_q6',
                       'multiplicity': 1,
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'PauliString(IIZIIIIII)',
                                       'operation': 'declare_readout_observable',
                                       'targets': ['q6']}]},
                      {'hamiltonian_coefficient': '-bare_coupling_g^(-2)/lattice_spacing_a',
                       'id': 'op_b_z_q7',
                       'multiplicity': 1,
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'PauliString(IZIIIIIII)',
                                       'operation': 'declare_readout_observable',
                                       'targets': ['q7']}]},
                      {'hamiltonian_coefficient': '-bare_coupling_g^(-2)/lattice_spacing_a',
                       'id': 'op_b_z_q8',
                       'multiplicity': 1,
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'PauliString(ZIIIIIIII)',
                                       'operation': 'declare_readout_observable',
                                       'targets': ['q8']}]}],
 'reconstruction': {'output_components': ['real', 'imag'],
                    'rule': 'Require sample_values to equal the authoritative empty object {} '
                            'and reject any nonempty, missing, or extra assignment. Let e[id] '
                            'be the scalar expectation returned for the exact recipe_id. The '
                            'real output is the sum of all listed coefficient*e[recipe_id] '
                            'terms. The imaginary output is exactly 0 because both requested '
                            'matrix elements are diagonal expectations of the Hermitian H_qc. '
                            'No identity shift or state-dependent Hamiltonian coefficient is '
                            'added.',
                    'terms': [{'coefficient': '-2*bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'glueball_e_x_q0_m2_real'},
                              {'coefficient': '-bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'glueball_e_x_q5_q0_real'},
                              {'coefficient': '-bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'glueball_e_x_q6_q3_real'},
                              {'coefficient': '-2*bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'glueball_e_x_q6_m2_real'},
                              {'coefficient': '-bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'glueball_e_x_q1_real'},
                              {'coefficient': '-bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'glueball_e_x_q4_q1_real'},
                              {'coefficient': '-bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'glueball_e_x_q7_q4_real'},
                              {'coefficient': '-bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'glueball_e_x_q7_real'},
                              {'coefficient': '-2*bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'glueball_e_x_q2_m2_real'},
                              {'coefficient': '-bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'glueball_e_x_q5_q2_real'},
                              {'coefficient': '-bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'glueball_e_x_q8_q5_real'},
                              {'coefficient': '-2*bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'glueball_e_x_q8_m2_real'},
                              {'coefficient': '-bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'glueball_e_x_q3_real'},
                              {'coefficient': '-bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'glueball_e_x_q1_q0_real'},
                              {'coefficient': '-bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'glueball_e_x_q4_q3_real'},
                              {'coefficient': '-bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'glueball_e_x_q7_q6_real'},
                              {'coefficient': '-bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'glueball_e_x_q2_q1_real'},
                              {'coefficient': '-bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'glueball_e_x_q5_q4_real'},
                              {'coefficient': '-bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'glueball_e_x_q8_q7_real'},
                              {'coefficient': '-bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'glueball_e_x_q5_real'},
                              {'coefficient': '-bare_coupling_g^(-2)/lattice_spacing_a',
                               'recipe_id': 'glueball_b_z_q0_real'},
                              {'coefficient': '-bare_coupling_g^(-2)/lattice_spacing_a',
                               'recipe_id': 'glueball_b_z_q1_real'},
                              {'coefficient': '-bare_coupling_g^(-2)/lattice_spacing_a',
                               'recipe_id': 'glueball_b_z_q2_real'},
                              {'coefficient': '-bare_coupling_g^(-2)/lattice_spacing_a',
                               'recipe_id': 'glueball_b_z_q3_real'},
                              {'coefficient': '-bare_coupling_g^(-2)/lattice_spacing_a',
                               'recipe_id': 'glueball_b_z_q4_real'},
                              {'coefficient': '-bare_coupling_g^(-2)/lattice_spacing_a',
                               'recipe_id': 'glueball_b_z_q5_real'},
                              {'coefficient': '-bare_coupling_g^(-2)/lattice_spacing_a',
                               'recipe_id': 'glueball_b_z_q6_real'},
                              {'coefficient': '-bare_coupling_g^(-2)/lattice_spacing_a',
                               'recipe_id': 'glueball_b_z_q7_real'},
                              {'coefficient': '-bare_coupling_g^(-2)/lattice_spacing_a',
                               'recipe_id': 'glueball_b_z_q8_real'},
                              {'coefficient': '+2*bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'vacuum_e_x_q0_m2_real'},
                              {'coefficient': '+bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'vacuum_e_x_q5_q0_real'},
                              {'coefficient': '+bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'vacuum_e_x_q6_q3_real'},
                              {'coefficient': '+2*bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'vacuum_e_x_q6_m2_real'},
                              {'coefficient': '+bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'vacuum_e_x_q1_real'},
                              {'coefficient': '+bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'vacuum_e_x_q4_q1_real'},
                              {'coefficient': '+bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'vacuum_e_x_q7_q4_real'},
                              {'coefficient': '+bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'vacuum_e_x_q7_real'},
                              {'coefficient': '+2*bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'vacuum_e_x_q2_m2_real'},
                              {'coefficient': '+bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'vacuum_e_x_q5_q2_real'},
                              {'coefficient': '+bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'vacuum_e_x_q8_q5_real'},
                              {'coefficient': '+2*bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'vacuum_e_x_q8_m2_real'},
                              {'coefficient': '+bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'vacuum_e_x_q3_real'},
                              {'coefficient': '+bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'vacuum_e_x_q1_q0_real'},
                              {'coefficient': '+bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'vacuum_e_x_q4_q3_real'},
                              {'coefficient': '+bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'vacuum_e_x_q7_q6_real'},
                              {'coefficient': '+bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'vacuum_e_x_q2_q1_real'},
                              {'coefficient': '+bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'vacuum_e_x_q5_q4_real'},
                              {'coefficient': '+bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'vacuum_e_x_q8_q7_real'},
                              {'coefficient': '+bare_coupling_g^2/lattice_spacing_a',
                               'recipe_id': 'vacuum_e_x_q5_real'},
                              {'coefficient': '+bare_coupling_g^(-2)/lattice_spacing_a',
                               'recipe_id': 'vacuum_b_z_q0_real'},
                              {'coefficient': '+bare_coupling_g^(-2)/lattice_spacing_a',
                               'recipe_id': 'vacuum_b_z_q1_real'},
                              {'coefficient': '+bare_coupling_g^(-2)/lattice_spacing_a',
                               'recipe_id': 'vacuum_b_z_q2_real'},
                              {'coefficient': '+bare_coupling_g^(-2)/lattice_spacing_a',
                               'recipe_id': 'vacuum_b_z_q3_real'},
                              {'coefficient': '+bare_coupling_g^(-2)/lattice_spacing_a',
                               'recipe_id': 'vacuum_b_z_q4_real'},
                              {'coefficient': '+bare_coupling_g^(-2)/lattice_spacing_a',
                               'recipe_id': 'vacuum_b_z_q5_real'},
                              {'coefficient': '+bare_coupling_g^(-2)/lattice_spacing_a',
                               'recipe_id': 'vacuum_b_z_q6_real'},
                              {'coefficient': '+bare_coupling_g^(-2)/lattice_spacing_a',
                               'recipe_id': 'vacuum_b_z_q7_real'},
                              {'coefficient': '+bare_coupling_g^(-2)/lattice_spacing_a',
                               'recipe_id': 'vacuum_b_z_q8_real'}]},
 'schema_version': '1.1',
 'state_labels': ['glueball', 'vacuum'],
 'state_preparation_interface': {'required_operations': ['prepare_glueball', 'prepare_vacuum'],
                                 'treat_as_black_box': True},
 'time_evolution': {'method': 'none: direct equal-time Pauli expectation measurement',
                    'recommended_test_steps': [],
                    'required': False,
                    'step_operations': [],
                    'step_symbol': 'N_T',
                    'symmetry_preservation_rule': 'The runtime passes the sentinel N_T=0 and '
                                                  'no evolution is constructed. The supplied '
                                                  'state preparations must already satisfy '
                                                  'every mandatory G_v=+1 constraint. Direct '
                                                  'Pauli readout introduces no finite-step '
                                                  'symmetry breaking.'}}

_RECONSTRUCTION_WEIGHTS = {}


def configure_runtime(parameter_contract, observable_specification):
    global _RECONSTRUCTION_WEIGHTS

    if not isinstance(parameter_contract, dict):
        raise TypeError("parameter_contract must be a dict")
    if not isinstance(observable_specification, dict):
        raise TypeError("observable_specification must be a dict")

    if parameter_contract.get("schema_version") != "1.1":
        raise ValueError("Unsupported parameter-contract schema")
    model = parameter_contract.get("model")
    if not isinstance(model, dict) or model.get("name") != PARAMETER_CONTRACT["model"]["name"]:
        raise ValueError("Parameter contract selects the wrong model")
    if model.get("spacetime_dimension") != "2+1":
        raise ValueError("Expected a 2+1 dimensional model")
    if model.get("gauge_structure") != "local Z2":
        raise ValueError("Expected local Z2 gauge structure")

    discretizations = parameter_contract.get("discretizations")
    if not isinstance(discretizations, list) or len(discretizations) != 2:
        raise ValueError("Expected the two authoritative discretizations")
    spacetime = discretizations[0]
    gauge = discretizations[1]
    if spacetime.get("sector") != "spacetime":
        raise ValueError("Missing spacetime discretization")
    spacetime_settings = spacetime.get("settings", {})
    if spacetime_settings.get("spatial_shape") != [4, 4]:
        raise ValueError("Only the 4x4 lattice is supported")
    if spacetime_settings.get("spatial_boundary_condition") != "open":
        raise ValueError("Only open spatial boundaries are supported")
    if spacetime_settings.get("vertex_count") != 16:
        raise ValueError("Expected 16 vertices")
    if spacetime_settings.get("link_count") != 24:
        raise ValueError("Expected 24 links")
    if spacetime_settings.get("plaquette_count") != 9:
        raise ValueError("Expected 9 plaquettes")
    if spacetime_settings.get("time_realization") != "continuous":
        raise ValueError("Expected continuous Hamiltonian time")

    if gauge.get("sector") != "gauge":
        raise ValueError("Missing gauge discretization")
    gauge_settings = gauge.get("settings", {})
    if gauge_settings.get("gauge_group") != "Z2":
        raise ValueError("Expected Z2 gauge group")
    if gauge_settings.get("gauss_sector") != "G_v=+1 at every lattice vertex":
        raise ValueError("Expected the charge-free G_v=+1 sector")
    if gauge_settings.get("effective_qubit_count") != 9:
        raise ValueError("Expected exactly nine reduced qubits")
    if gauge_settings.get("physical_dimension") != 512:
        raise ValueError("Expected the complete 512-dimensional reduced space")

    external_states = parameter_contract.get("external_states")
    if not isinstance(external_states, dict):
        raise ValueError("Missing external-state contract")
    if external_states.get("labels") != ["glueball", "vacuum"]:
        raise ValueError("External state labels or ordering do not match")
    definitions = external_states.get("definitions", {})
    if "level_index=1" not in definitions.get("glueball", ""):
        raise ValueError("Glueball must retain zero-based level_index=1")
    if definitions.get("vacuum") != "Ground state / vacuum":
        raise ValueError("Vacuum state definition does not match")

    parameters = parameter_contract.get("parameters")
    if not isinstance(parameters, list):
        raise ValueError("Missing parameter list")
    selected = {}
    for parameter in parameters:
        if not isinstance(parameter, dict) or "name" not in parameter:
            raise ValueError("Malformed parameter entry")
        selected[parameter["name"]] = parameter.get("selected_value")

    if selected.get("number of lattice site [L,L]:") != [4, 4]:
        raise ValueError("Immutable lattice-site input must be [4, 4]")
    if not isinstance(selected.get("bare copuling: g"), str):
        raise ValueError("Missing immutable bare-coupling user input")

    g = selected.get("bare_coupling_g")
    a = selected.get("lattice_spacing_a")
    if isinstance(g, bool) or not isinstance(g, Real) or not math.isfinite(float(g)):
        raise TypeError("bare_coupling_g must be a finite real number")
    if isinstance(a, bool) or not isinstance(a, Real) or not math.isfinite(float(a)):
        raise TypeError("lattice_spacing_a must be a finite real number")
    g = float(g)
    a = float(a)
    if g <= 0.0 or a <= 0.0:
        raise ValueError("bare_coupling_g and lattice_spacing_a must be positive")

    if observable_specification.get("observable_schema_version") != "1.0":
        raise ValueError("Unsupported observable-specification schema")
    if observable_specification.get("name") != "user_observable":
        raise ValueError("Observable name does not match")
    if observable_specification.get("expression") != "Hamiltonian: H":
        raise ValueError("Only the requested Hamiltonian observable is supported")
    if observable_specification.get("ordering") != "as_written":
        raise ValueError("Observable ordering must be as_written")
    if observable_specification.get("insertions") != [
        {"id": "Hamiltonian", "operator": "H", "position": "global"}
    ]:
        raise ValueError("Observable insertion structure does not match")
    if observable_specification.get("matrix_elements") != [
        {
            "id": "glueball energy",
            "bra": "glueball",
            "ket": "glueball",
            "coefficient": 1.0,
        },
        {
            "id": "vacuum energy",
            "bra": "vacuum",
            "ket": "vacuum",
            "coefficient": -1.0,
        },
    ]:
        raise ValueError("Observable matrix elements do not match")
    if observable_specification.get("requested_components") != ["real", "imag"]:
        raise ValueError("Observable components do not match")
    if observable_specification.get("sampling") != {
        "mode": "explicit_points",
        "points": [{"sample_id": "default", "values": {}}],
    }:
        raise ValueError("Only the authoritative empty sample assignment is supported")

    electric_scale = (g * g) / a
    magnetic_scale = 1.0 / (g * g * a)
    weights = {}
    for state in ("glueball", "vacuum"):
        state_sign = -1.0 if state == "glueball" else 1.0
        for operator_id, _, sector, multiplicity in _OPERATOR_DATA:
            scale = electric_scale if sector == "electric" else magnetic_scale
            weights[f"{state}_{operator_id}_real"] = (
                state_sign * float(multiplicity) * scale
            )

    _RECONSTRUCTION_WEIGHTS = weights


def build_observable_measurements(state_circuits, trotter_steps, sample_values):
    if not isinstance(state_circuits, dict):
        raise TypeError("state_circuits must be a dict")
    if set(state_circuits) != {"glueball", "vacuum"}:
        raise ValueError(
            "state_circuits must contain exactly 'glueball' and 'vacuum'"
        )
    if isinstance(trotter_steps, bool) or not isinstance(trotter_steps, int):
        raise TypeError("trotter_steps must be the integer sentinel 0")
    if trotter_steps != 0:
        raise ValueError("This protocol requires trotter_steps == 0")
    if not isinstance(sample_values, dict) or sample_values != {}:
        raise ValueError("sample_values must be exactly the empty assignment {}")

    for state in ("glueball", "vacuum"):
        circuit = state_circuits[state]
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError(f"state_circuits[{state!r}] must be a QuantumCircuit")
        if circuit.num_qubits != 9:
            raise ValueError(f"state_circuits[{state!r}] must have exactly 9 qubits")
        if circuit.num_clbits != 0:
            raise ValueError(
                f"state_circuits[{state!r}] must contain no classical bits"
            )
        if circuit.parameters:
            raise ValueError(
                f"state_circuits[{state!r}] must be fully parameter-bound"
            )
        for instruction in circuit.data:
            operation = (
                instruction.operation
                if hasattr(instruction, "operation")
                else instruction[0]
            )
            if operation.name in {"measure", "reset", "initialize"}:
                raise ValueError(
                    f"state_circuits[{state!r}] contains forbidden operation "
                    f"{operation.name!r}"
                )
            if getattr(operation, "condition", None) is not None:
                raise ValueError(
                    f"state_circuits[{state!r}] contains classical control"
                )

    recipes = []
    for state in ("glueball", "vacuum"):
        source = state_circuits[state]
        for operator_id, pauli, _, _ in _OPERATOR_DATA:
            circuit = source.copy()
            if circuit.parameters or circuit.num_clbits != 0:
                raise RuntimeError("Copied preparation circuit violated protocol invariants")
            recipes.append(
                {
                    "recipe_id": f"{state}_{operator_id}_real",
                    "component": "real",
                    "circuit": circuit,
                    "readout_pauli": pauli,
                }
            )

    if tuple(recipe["recipe_id"] for recipe in recipes) != _RECIPE_IDS:
        raise RuntimeError("Internal recipe ordering does not match the contract")
    return recipes


def reconstruct_observable(expectations, sample_values):
    if not isinstance(sample_values, dict) or sample_values != {}:
        raise ValueError("sample_values must be exactly the empty assignment {}")
    if not isinstance(expectations, dict):
        raise TypeError("expectations must be a dict")
    if set(expectations) != set(_RECIPE_IDS):
        missing = sorted(set(_RECIPE_IDS) - set(expectations))
        extra = sorted(set(expectations) - set(_RECIPE_IDS))
        raise ValueError(
            f"expectations must contain every exact recipe_id; "
            f"missing={missing}, extra={extra}"
        )

    total = 0.0
    for recipe_id in _RECIPE_IDS:
        value = expectations[recipe_id]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"Expectation for {recipe_id!r} must be a real number")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"Expectation for {recipe_id!r} must be finite")
        total += _RECONSTRUCTION_WEIGHTS[recipe_id] * value

    return complex(total, 0.0)


configure_runtime(PARAMETER_CONTRACT, OBSERVABLE_SPECIFICATION)

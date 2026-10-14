import numpy as np
from fractions import Fraction

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import XXPlusYYGate, RZZGate, RZGate
from qiskit.quantum_info import SparsePauliOp


PARAMETER_CONTRACT = {'constraints': [{'applies_when': 'always',
                  'id': 'equal_temporal_spatial_spacing',
                  'parameters': ['temporal_lattice_spacing', 'lattice_spacing'],
                  'reason': 'The continuum contract requires equal temporal and spatial '
                            'spacing.',
                  'requirement': 'temporal_lattice_spacing must equal lattice_spacing',
                  'severity': 'fatal'},
                 {'applies_when': 'always',
                  'id': 'open_boundary_gauss_reconstruction',
                  'parameters': ['bare_gauge_coupling'],
                  'reason': 'The ordered fermion insertions visit an intermediate '
                            'charge-minus-one sector.',
                  'requirement': 'Use the declared reflection-symmetric open-boundary '
                                 'electric-field reconstruction in every total-charge sector '
                                 'visited by the ordered observable.',
                  'severity': 'fatal'},
                 {'applies_when': 'always',
                  'id': 'preserve_unshifted_hamiltonian',
                  'parameters': [],
                  'reason': 'The requested lattice Hamiltonian is the direct unshifted '
                            'discretization.',
                  'requirement': 'Do not subtract any vacuum or target-state energy and do not '
                                 'introduce state-dependent Hamiltonian coefficients.',
                  'severity': 'fatal'},
                 {'applies_when': 'always',
                  'id': 'fixed_gamma_convention',
                  'parameters': ['lightcone_normalization'],
                  'reason': 'These continuum conventions are immutable.',
                  'requirement': 'Use gamma_0=sigma_1, gamma_1=-i*sigma_2, alpha=sigma_3, '
                                 'beta=sigma_1 and gamma_minus=gamma_0-gamma_1 with '
                                 'normalization one.',
                  'severity': 'fatal'},
                 {'applies_when': 'always',
                  'id': 'observable_range_fit',
                  'parameters': ['lattice_spacing'],
                  'reason': 'No requested observable sample may be dropped.',
                  'requirement': 'The site map s(z)=z+2 must be valid for every supplied z, '
                                 'including both endpoints.',
                  'severity': 'fatal'},
                 {'applies_when': 'always',
                  'id': 'no_gauge_flux_truncation',
                  'parameters': [],
                  'reason': 'Reintroducing links would change the selected finite '
                            'representation.',
                  'requirement': 'Do not introduce independent gauge-link variables or an '
                                 'electric-flux cutoff after exact Gauss-law elimination.',
                  'severity': 'fatal'}],
 'discretizations': [{'parameter_references': ['lattice_spacing', 'temporal_lattice_spacing'],
                      'scheme': 'centered one-dimensional spatial lattice with equal temporal '
                                'spacing',
                      'sector': 'spacetime',
                      'settings': {'gauss_relation_count': 5,
                                   'observable_site_map': 's(z)=z+2',
                                   'plaquette_count': 0,
                                   'spatial_boundary_condition': 'open',
                                   'spatial_coordinate': 'x_N=a*(N-2)',
                                   'spatial_dimension': 1,
                                   'spatial_link_count': 4,
                                   'spatial_link_index_range': 'ell=0,...,3',
                                   'spatial_site_count': 5,
                                   'spatial_site_index_range': 'N=0,...,4',
                                   'temporal_spacing_relation': 'a_t=a'},
                      'truncation': {'reason': 'The immutable five-site lattice supports every '
                                               'requested separation.',
                                     'scheme': 'finite spatial volume',
                                     'settings': {'coordinate_interval': '[-2a,2a]'}}},
                     {'parameter_references': ['bare_fermion_mass',
                                               'wilson_parameter',
                                               'charge_reference_occupancy'],
                      'scheme': 'two-component Wilson fermions',
                      'sector': 'fermion',
                      'settings': {'alpha': 'sigma_3',
                                   'beta': 'sigma_1',
                                   'charge_density': 'q_N=sum_alpha '
                                                     'c_dagger[N,alpha]*c[N,alpha]-1',
                                   'component_count_per_site': 2,
                                   'fermionic_mode_count': 10,
                                   'gamma_0': 'sigma_1',
                                   'gamma_1': '-i*sigma_2',
                                   'wilson_parameter': 1},
                      'truncation': {'reason': 'Fermionic occupation is intrinsically finite '
                                               'and is not an approximate field truncation.',
                                     'scheme': 'exact fermionic occupation',
                                     'settings': {'local_mode_occupations': [0, 1],
                                                  'neutral_sector_dimension': 252,
                                                  'total_fock_dimension': 1024}}},
                     {'parameter_references': ['bare_gauge_coupling'],
                      'scheme': 'spatial axial gauge with exact open-boundary Gauss-law '
                                'elimination',
                      'sector': 'gauge',
                      'settings': {'effective_degree_count': 10,
                                   'effective_qubit_count': 10,
                                   'electric_field_reconstruction': 'E_-1=-(g/2)*Q_lat; '
                                                                    'E_ell=g*(S_ell-Q_lat/2) '
                                                                    'for ell=0,...,3; '
                                                                    'E_4=(g/2)*Q_lat',
                                   'external_neutral_sector_dimension': 252,
                                   'gauge_condition': 'A_1=0',
                                   'gauss_identity': 'E_N-E_{N-1}=g*q_N at every lattice '
                                                     'vertex',
                                   'independent_constraint_count': 4,
                                   'independent_gauge_degree_count': 0,
                                   'kinematic_dimension_after_reduction': 1024,
                                   'locality_change': 'Local link-electric energy becomes a '
                                                      'long-range fermionic Coulomb '
                                                      'interaction.',
                                   'partial_charge': 'S_ell=sum_{N=0}^{ell} q_N',
                                   'physical_dimension_with_all_required_charge_sectors': 1024,
                                   'scalar_potential_reconstruction': 'A0_N=a*g*sum_{ell=0}^{3}(indicator(N<=ell)-1/2)*(S_ell-Q_lat/2)',
                                   'total_charge': 'Q_lat=sum_{N=0}^{4} q_N'},
                      'truncation': {'reason': 'No independent infinite-dimensional gauge '
                                               'Hilbert space remains after exact Gauss-law '
                                               'elimination.',
                                     'scheme': 'none',
                                     'settings': {'electric_flux_cutoff': 'none',
                                                  'gauge_links': 'exactly eliminated'}}}],
 'external_states': {'definitions': {'phi': 'Quantum-number-sector eigenstate; level_index=0; '
                                            'quantum_numbers={"charge": 0}',
                                     'psi': 'Quantum-number-sector eigenstate; level_index=1; '
                                            'quantum_numbers={"charge": 0}'},
                     'equivalent_groups': [],
                     'labels': ['phi', 'psi']},
 'model': {'gauge_structure': 'local U(1)',
           'name': 'Schwinger model',
           'spacetime_dimension': '1+1'},
 'parameters': [{'domain': {'minimum': 0},
                 'name': 'bare_gauge_coupling',
                 'required_for_run': True,
                 'role': 'Bare U(1) gauge coupling.',
                 'selected_value': 0.5,
                 'symbol': 'g',
                 'type': 'real',
                 'unit': 'inverse_length'},
                {'domain': {'rule': 'finite real number'},
                 'name': 'bare_fermion_mass',
                 'required_for_run': True,
                 'role': 'Bare Dirac fermion mass.',
                 'selected_value': 0.5,
                 'symbol': 'm',
                 'type': 'real',
                 'unit': 'inverse_length'},
                {'domain': {'exclusive_minimum': 0},
                 'name': 'lattice_spacing',
                 'required_for_run': True,
                 'role': 'Spatial lattice spacing.',
                 'selected_value': 1.0,
                 'symbol': 'a',
                 'type': 'real',
                 'unit': 'length'},
                {'domain': {'exclusive_minimum': 0},
                 'name': 'temporal_lattice_spacing',
                 'required_for_run': True,
                 'role': 'Temporal spacing, fixed equal to the spatial spacing.',
                 'selected_value': 1.0,
                 'symbol': 'a_t',
                 'type': 'real',
                 'unit': 'time'},
                {'domain': {'rule': 'nonzero finite real number'},
                 'name': 'wilson_parameter',
                 'required_for_run': True,
                 'role': 'Wilson term coefficient used to remove fermion doublers.',
                 'selected_value': 1.0,
                 'symbol': 'r',
                 'type': 'real',
                 'unit': None},
                {'domain': {'allowed_values': [1]},
                 'name': 'charge_reference_occupancy',
                 'required_for_run': True,
                 'role': 'Fixed state-independent occupancy reference in q_N=c_N^dagger '
                         'c_N-n_ref.',
                 'selected_value': 1,
                 'symbol': 'n_ref',
                 'type': 'integer',
                 'unit': None},
                {'domain': {'allowed_values': [1.0]},
                 'name': 'lightcone_normalization',
                 'required_for_run': True,
                 'role': 'Normalization in gamma^-=gamma^0-gamma^1 and X(z)=az(1,1).',
                 'selected_value': 1.0,
                 'symbol': 'c_lc',
                 'type': 'real',
                 'unit': None},
                {'domain': {'source': 'immutable RunSpec value'},
                 'immutable': True,
                 'name': 'bare coupling: g',
                 'required_for_run': True,
                 'role': 'Immutable user input; downstream stages must reference this path.',
                 'selected_value': 0.5,
                 'source_ref': '/user_input/parameters/bare coupling: g',
                 'symbol': 'bare coupling: g',
                 'type': 'real',
                 'unit': None},
                {'domain': {'source': 'immutable RunSpec value'},
                 'immutable': True,
                 'name': 'bare quark mass: m',
                 'required_for_run': True,
                 'role': 'Immutable user input; downstream stages must reference this path.',
                 'selected_value': 0.5,
                 'source_ref': '/user_input/parameters/bare quark mass: m',
                 'symbol': 'bare quark mass: m',
                 'type': 'real',
                 'unit': None},
                {'domain': {'source': 'immutable RunSpec value'},
                 'immutable': True,
                 'name': 'lattice spacing: a',
                 'required_for_run': True,
                 'role': 'Immutable user input; downstream stages must reference this path.',
                 'selected_value': 1.0,
                 'source_ref': '/user_input/parameters/lattice spacing: a',
                 'symbol': 'lattice spacing: a',
                 'type': 'real',
                 'unit': None},
                {'domain': {'source': 'immutable RunSpec value'},
                 'immutable': True,
                 'name': 'number of lattice site: L',
                 'required_for_run': True,
                 'role': 'Immutable user input; downstream stages must reference this path.',
                 'selected_value': 5,
                 'source_ref': '/user_input/parameters/number of lattice site: L',
                 'symbol': 'number of lattice site: L',
                 'type': 'integer',
                 'unit': None}],
 'schema_version': '1.1',
 'validation_obligations': [{'description': 'Confirm that phi is the zero-based level 0 '
                                            'eigenstate and psi is the zero-based level 1 '
                                            'eigenstate of H_lat within the Q_lat=0 sector, '
                                            'without identifying the two states.',
                             'id': 'external_state_sector_and_level_check',
                             'severity': 'fatal',
                             'stage': 'post_construction'},
                            {'description': 'Confirm that E_N-E_{N-1}-g*q_N vanishes as an '
                                            'operator identity at every lattice vertex under '
                                            'the declared reconstruction, including the '
                                            'intermediate charge-minus-one sector.',
                             'id': 'physical_gauss_sector_check',
                             'severity': 'fatal',
                             'stage': 'post_construction'},
                            {'description': 'Confirm that [H_lat,Q_lat]=0 in the full '
                                            '1024-dimensional reduced Fock space.',
                             'id': 'charge_commutator_check',
                             'severity': 'fatal',
                             'stage': 'post_construction'},
                            {'description': 'Confirm the ordered path Q=0 -> Q=-1 -> Q=-1 -> '
                                            'Q=0 for the right-to-left action of psi '
                                            'insertion, Wilson transporter, and bar-psi '
                                            'insertion.',
                             'id': 'ordered_sector_path_check',
                             'severity': 'fatal',
                             'stage': 'pre_simulation'},
                            {'description': 'Confirm that the constructed Hamiltonian contains '
                                            'no additive vacuum-energy subtraction or '
                                            'target-state-dependent coefficient.',
                             'id': 'no_energy_shift_check',
                             'severity': 'fatal',
                             'stage': 'post_construction'}]}


STATE_PREPARATION_CONTRACT = {'adapter_interface': {'builder': 'build_state_preparation',
                       'returns': ['QuantumCircuit', 'ordered_parameters'],
                       'supports_controlled': True,
                       'supports_inverse': True,
                       'unitary': True},
 'ansatz': {'depth_extension_rule': "Prepare each label's reference once and apply complete "
                                    'shared-parameter layers ell=1,...,D_ans in increasing '
                                    'ell. Extend D_ans to D_ans+1 by appending one complete '
                                    'layer with fresh theta[D_ans+1,j], zeta[D_ans+1,j], and '
                                    'rho[D_ans+1,j]. The earlier gate sequence is an exact '
                                    'prefix. Setting every appended angle to 0 makes every '
                                    'appended gate exactly the identity.',
            'depth_symbol': 'D_ans',
            'family_id': 'wilson_schwinger_qc10_shared_local_number_conserving_xxyy_zz_z',
            'layer_operations': [{'angle': '2*theta[ell,j] in the Qiskit XXPlusYYGate angle '
                                           'convention',
                                  'controls': [],
                                  'gate': 'Qiskit XXPlusYYGate(2*theta[ell,j],0), exactly '
                                          'exp[-i*theta[ell,j]*(X_j*X_(j+1)+Y_j*Y_(j+1))/2]',
                                  'index_ranges': {'ell': '1,...,D_ans in increasing order',
                                                   'iteration_order': 'Within each layer apply '
                                                                      'all XXPlusYYGate '
                                                                      'instances first, '
                                                                      'sequentially in '
                                                                      'increasing j',
                                                   'j': '0,...,8 in increasing order',
                                                   'parameter_sharing': 'theta[ell,j] is '
                                                                        'shared between the '
                                                                        'phi and psi circuit '
                                                                        'instances'},
                                  'operation': 'gate_or_indexed_loop',
                                  'targets': ['q_j', 'q_(j+1)']},
                                 {'angle': 'zeta[ell,j]',
                                  'controls': [],
                                  'gate': 'Qiskit RZZGate(zeta[ell,j]) with '
                                          'RZZ(alpha)=exp(-i*alpha*Z_j*Z_(j+1)/2)',
                                  'index_ranges': {'ell': '1,...,D_ans in increasing order',
                                                   'iteration_order': 'After all XXPlusYYGate '
                                                                      'instances of layer ell, '
                                                                      'apply all RZZGate '
                                                                      'instances sequentially '
                                                                      'in increasing j',
                                                   'j': '0,...,8 in increasing order',
                                                   'parameter_sharing': 'zeta[ell,j] is shared '
                                                                        'between the phi and '
                                                                        'psi circuit '
                                                                        'instances'},
                                  'operation': 'gate_or_indexed_loop',
                                  'targets': ['q_j', 'q_(j+1)']},
                                 {'angle': 'rho[ell,j]',
                                  'controls': [],
                                  'gate': 'Qiskit RZGate(rho[ell,j]) with '
                                          'RZ(alpha)=exp(-i*alpha*Z_j/2)',
                                  'index_ranges': {'ell': '1,...,D_ans in increasing order',
                                                   'iteration_order': 'After all RZZGate '
                                                                      'instances of layer ell, '
                                                                      'apply RZGate instances '
                                                                      'in increasing j',
                                                   'j': '0,...,9 in increasing order',
                                                   'parameter_sharing': 'rho[ell,j] is shared '
                                                                        'between the phi and '
                                                                        'psi circuit '
                                                                        'instances'},
                                  'operation': 'gate_or_indexed_loop',
                                  'targets': ['q_j']}],
            'nested_by_depth': True,
            'parameter_order_rule': 'Use one shared parameter vector for phi and psi. Order '
                                    'layers by ell=1,...,D_ans. Within layer ell, first list '
                                    'theta[ell,j] for j=0,...,8 in increasing j, then '
                                    'zeta[ell,j] for j=0,...,8 in increasing j, then '
                                    'rho[ell,j] for j=0,...,9 in increasing j. Distinct '
                                    'indices and layers share no parameter. Each layer has 9 '
                                    'theta, 9 zeta, and 10 rho parameters, totaling 28 '
                                    'independent shared parameters.',
            'physical_sector': {'preserved_exactly': True,
                                'required': True,
                                'rule': 'Both references have Hamming weight 5 and '
                                        'Q_lat=-(1/2)*sum_j Z_j=0. Every adjacent XXPlusYY '
                                        'gate mixes only |01> and |10> and commutes with '
                                        'n_j+n_(j+1); every RZZ and RZ gate is occupation '
                                        'diagonal. Therefore every intermediate gate, complete '
                                        'ordered layer, and repeated-layer circuit remains in '
                                        'the full 252-dimensional weight-5 sector. The '
                                        'authoritative reconstructed G_0,...,G_4 Pauli sums '
                                        'cancel identically on the full ten-qubit space, so '
                                        'the mapped Gauss constraints are also preserved '
                                        'exactly. Adjacent exchanges connect the complete '
                                        'weight-5 configuration graph, while RZZ and RZ phases '
                                        'provide non-Gaussian interacting phases.'},
            'recommended_test_depths': [1, 2, 3, 4],
            'reference_state_rule': 'Use two mutually orthogonal weight-5 computational '
                                    'references. For phi initialize |0>^10 and apply X to q1, '
                                    'q3, q5, q7, q9. For psi initialize |0>^10 and apply X to '
                                    'q1, q3, q4, q7, q9. Apply the same parameterized unitary '
                                    'to both references, so their orthogonality is preserved '
                                    'exactly while they represent distinct requested levels.',
            'reference_states': {'phi': {'basis_state': '0101010101 in declared logical order '
                                                        'q0=c[0,1], q1=c[0,2], q2=c[1,1], '
                                                        'q3=c[1,2], q4=c[2,1], q5=c[2,2], '
                                                        'q6=c[3,1], q7=c[3,2], q8=c[4,1], '
                                                        'q9=c[4,2]; equivalently |1010101010> '
                                                        'in Qiskit b9...b0 display order',
                                         'operations': [{'angle': 'not_applicable',
                                                         'controls': [],
                                                         'gate': 'XGate',
                                                         'operation': 'gate',
                                                         'targets': ['q1_mode_c[0,2]']},
                                                        {'angle': 'not_applicable',
                                                         'controls': [],
                                                         'gate': 'XGate',
                                                         'operation': 'gate',
                                                         'targets': ['q3_mode_c[1,2]']},
                                                        {'angle': 'not_applicable',
                                                         'controls': [],
                                                         'gate': 'XGate',
                                                         'operation': 'gate',
                                                         'targets': ['q5_mode_c[2,2]']},
                                                        {'angle': 'not_applicable',
                                                         'controls': [],
                                                         'gate': 'XGate',
                                                         'operation': 'gate',
                                                         'targets': ['q7_mode_c[3,2]']},
                                                        {'angle': 'not_applicable',
                                                         'controls': [],
                                                         'gate': 'XGate',
                                                         'operation': 'gate',
                                                         'targets': ['q9_mode_c[4,2]']}]},
                                 'psi': {'basis_state': '0101100101 in declared logical order '
                                                        'q0=c[0,1], q1=c[0,2], q2=c[1,1], '
                                                        'q3=c[1,2], q4=c[2,1], q5=c[2,2], '
                                                        'q6=c[3,1], q7=c[3,2], q8=c[4,1], '
                                                        'q9=c[4,2]; equivalently |1010011010> '
                                                        'in Qiskit b9...b0 display order',
                                         'operations': [{'angle': 'not_applicable',
                                                         'controls': [],
                                                         'gate': 'XGate',
                                                         'operation': 'gate',
                                                         'targets': ['q1_mode_c[0,2]']},
                                                        {'angle': 'not_applicable',
                                                         'controls': [],
                                                         'gate': 'XGate',
                                                         'operation': 'gate',
                                                         'targets': ['q3_mode_c[1,2]']},
                                                        {'angle': 'not_applicable',
                                                         'controls': [],
                                                         'gate': 'XGate',
                                                         'operation': 'gate',
                                                         'targets': ['q4_mode_c[2,1]']},
                                                        {'angle': 'not_applicable',
                                                         'controls': [],
                                                         'gate': 'XGate',
                                                         'operation': 'gate',
                                                         'targets': ['q7_mode_c[3,2]']},
                                                        {'angle': 'not_applicable',
                                                         'controls': [],
                                                         'gate': 'XGate',
                                                         'operation': 'gate',
                                                         'targets': ['q9_mode_c[4,2]']}]}},
            'symmetry_preservation': [{'conserved_quantity_id': 'local_U1_gauss_law',
                                       'gate_level_reason': 'Under the authoritative '
                                                            'electric-field reconstruction '
                                                            'every expanded G_N is the zero '
                                                            'operator after exact Pauli '
                                                            'collection. Both references and '
                                                            'every number-conserving layer '
                                                            'therefore satisfy all five mapped '
                                                            'Gauss constraints without '
                                                            'gauge-link qubits or penalties.',
                                       'preserved_exactly': True,
                                       'state_labels': ['phi', 'psi'],
                                       'strategy': 'exact_circuit',
                                       'target_eigenvalues': {'phi': '0 at every lattice '
                                                                     'vertex',
                                                              'psi': '0 at every lattice '
                                                                     'vertex'}},
                                      {'conserved_quantity_id': 'vector_charge',
                                       'gate_level_reason': 'The references have five '
                                                            'occupations. XXPlusYY preserves '
                                                            'pair occupation and RZZ and RZ '
                                                            'are occupation diagonal, so every '
                                                            'gate commutes with Q_lat. '
                                                            'Adjacent exchanges connect all '
                                                            'weight-5 basis configurations, so '
                                                            'exact charge preservation does '
                                                            'not obstruct access to either '
                                                            'requested level.',
                                       'preserved_exactly': True,
                                       'state_labels': ['phi', 'psi'],
                                       'strategy': 'exact_circuit',
                                       'target_eigenvalues': {'phi': '0', 'psi': '0'}},
                                      {'conserved_quantity_id': 'C',
                                       'gate_level_reason': 'No charge-conjugation eigenvalue '
                                                            'is assigned. Generic local RZ and '
                                                            'RZZ parameters need not commute '
                                                            'with the particle-hole Clifford, '
                                                            'so imposing C would add an '
                                                            'unsupported restriction.',
                                       'preserved_exactly': False,
                                       'state_labels': ['phi', 'psi'],
                                       'strategy': 'diagnostic_only',
                                       'target_eigenvalues': {'phi': 'unspecified',
                                                              'psi': 'unspecified'}},
                                      {'conserved_quantity_id': 'P',
                                       'gate_level_reason': 'Fermionic reflection is an exact '
                                                            'Hamiltonian symmetry but neither '
                                                            'level has an assigned parity '
                                                            'eigenvalue. The ordered '
                                                            'open-chain layer is not '
                                                            'constrained to commute with the '
                                                            'FSWAP reflection.',
                                       'preserved_exactly': False,
                                       'state_labels': ['phi', 'psi'],
                                       'strategy': 'diagnostic_only',
                                       'target_eigenvalues': {'phi': 'unspecified',
                                                              'psi': 'unspecified'}},
                                      {'conserved_quantity_id': 'T',
                                       'gate_level_reason': 'T is antiunitary and has no '
                                                            'assigned target eigenvalue. '
                                                            'Generic exchange and phase angles '
                                                            'do not enforce the required FSWAP '
                                                            'component exchange followed by '
                                                            'complex conjugation.',
                                       'preserved_exactly': False,
                                       'state_labels': ['phi', 'psi'],
                                       'strategy': 'diagnostic_only',
                                       'target_eigenvalues': {'phi': 'unspecified',
                                                              'psi': 'unspecified'}},
                                      {'conserved_quantity_id': 'PT',
                                       'gate_level_reason': 'No PT eigenvalue is supplied, and '
                                                            'the generic shared layer is not '
                                                            'constrained to commute with the '
                                                            'declared antiunitary composition.',
                                       'preserved_exactly': False,
                                       'state_labels': ['phi', 'psi'],
                                       'strategy': 'diagnostic_only',
                                       'target_eigenvalues': {'phi': 'unspecified',
                                                              'psi': 'unspecified'}},
                                      {'conserved_quantity_id': 'spatial_translations',
                                       'gate_level_reason': 'The authoritative open-boundary '
                                                            'Hamiltonian explicitly breaks the '
                                                            'candidate cyclic spatial '
                                                            'translation, so translation is '
                                                            'not imposed on either reference '
                                                            'or layer.',
                                       'preserved_exactly': False,
                                       'state_labels': ['phi', 'psi'],
                                       'strategy': 'diagnostic_only',
                                       'target_eigenvalues': {'phi': 'unspecified',
                                                              'psi': 'unspecified'}},
                                      {'conserved_quantity_id': 'continuous_spacetime_translations',
                                       'gate_level_reason': 'There is no finite-lattice '
                                                            'continuous spatial-translation '
                                                            'generator. Requiring the '
                                                            'variational layer to commute with '
                                                            'the time-translation generator '
                                                            'H_qc would obstruct exploration '
                                                            'from the references and is not a '
                                                            'target-sector requirement.',
                                       'preserved_exactly': False,
                                       'state_labels': ['phi', 'psi'],
                                       'strategy': 'diagnostic_only',
                                       'target_eigenvalues': {'phi': 'unspecified',
                                                              'psi': 'unspecified'}},
                                      {'conserved_quantity_id': 'Lorentz_boost',
                                       'gate_level_reason': 'The contract supplies no exact '
                                                            'finite-lattice boost generator, '
                                                            'so no boost constraint is placed '
                                                            'on the circuit.',
                                       'preserved_exactly': False,
                                       'state_labels': ['phi', 'psi'],
                                       'strategy': 'diagnostic_only',
                                       'target_eigenvalues': {'phi': 'unspecified',
                                                              'psi': 'unspecified'}}]},
 'contract_type': 'state_preparation',
 'literature_sources': [{'applicability': 'modified: supports orthogonal same-sector '
                                          'references followed by a shared symmetry-preserving '
                                          'unitary, but its periodic-boundary Schwinger '
                                          'discretization and Hamiltonian groups do not match '
                                          'the authoritative open-boundary two-component '
                                          'Wilson Hamiltonian',
                         'locator': 'Section "Preparation of hadronic state", Eq. (refst) and '
                                    'the displayed definition U(theta)=product_j product_i '
                                    'exp(i Hbar_i theta_ij)',
                         'source_path': 'arXiv-2606.22602v1/GPDs_v2.tex'},
                        {'applicability': 'modified: provides same-model evidence for compact '
                                          'scalable symmetry-preserving circuits, but the '
                                          'supplied extract does not define a complete gate '
                                          'pool for the present Wilson encoding',
                         'locator': 'Section "Building Scalable State Preparation Quantum '
                                    'Circuits using SC-ADAPT-VQE with Classical Computing", '
                                    'Table tab:Echifmp5gp3 and Fig. fig:ClassConvXCirc',
                         'source_path': 'arXiv-2308.04481v3/Schwinger-v5.tex'},
                        {'applicability': 'incompatible: prepares localized hadron wave '
                                          'packets by adiabatic evolution rather than the two '
                                          'lowest eigenlevels of the authoritative '
                                          'open-boundary Wilson Hamiltonian',
                         'locator': 'Section "Hadron Wavepacket Preparation", Eq. '
                                    '(adiabaticWP)',
                         'source_path': 'arXiv-2401.08044v2/main_v6.tex'},
                        {'applicability': 'incompatible: defines a nonunitary mesonic '
                                          'wave-packet creation-operator ansatz for a periodic '
                                          'staggered lattice, not a unitary '
                                          'eigenstate-preparation layer for this encoding',
                         'locator': 'Section "Ansatz for interacting wave-packet creation '
                                    'operator", Eqs. (ansatz1) and (B-def)',
                         'source_path': 'arXiv-2402.00840v3/arxiv_v3_published.tex'}],
 'schema_version': '1.1',
 'state_labels': ['phi', 'psi']}


_HAMILTONIAN_TERMS = (
    ("IIIIIIIIII", "5/16"),
    ("IIIIIIIIXX", "3/4"),
    ("IIIIIIIIYY", "3/4"),
    ("IIIIIIXXII", "3/4"),
    ("IIIIIIYYII", "3/4"),
    ("IIIIXXIIII", "3/4"),
    ("IIIIYYIIII", "3/4"),
    ("IIXXIIIIII", "3/4"),
    ("IIYYIIIIII", "3/4"),
    ("XXIIIIIIII", "3/4"),
    ("YYIIIIIIII", "3/4"),
    ("IIIIIIIYZX", "1/4"),
    ("IIIIIIIXZY", "-1/4"),
    ("IIIIIIYZXI", "-1/4"),
    ("IIIIIIXZYI", "1/4"),
    ("IIIIIIXZZX", "-1/4"),
    ("IIIIIIYZZY", "-1/4"),
    ("IIIIIIIXXI", "-1/4"),
    ("IIIIIIIYYI", "-1/4"),
    ("IIIIIYZXII", "1/4"),
    ("IIIIIXZYII", "-1/4"),
    ("IIIIYZXIII", "-1/4"),
    ("IIIIXZYIII", "1/4"),
    ("IIIIXZZXII", "-1/4"),
    ("IIIIYZZYII", "-1/4"),
    ("IIIIIXXIII", "-1/4"),
    ("IIIIIYYIII", "-1/4"),
    ("IIIYZXIIII", "1/4"),
    ("IIIXZYIIII", "-1/4"),
    ("IIYZXIIIII", "-1/4"),
    ("IIXZYIIIII", "1/4"),
    ("IIXZZXIIII", "-1/4"),
    ("IIYZZYIIII", "-1/4"),
    ("IIIXXIIIII", "-1/4"),
    ("IIIYYIIIII", "-1/4"),
    ("IYZXIIIIII", "1/4"),
    ("IXZYIIIIII", "-1/4"),
    ("YZXIIIIIII", "-1/4"),
    ("XZYIIIIIII", "1/4"),
    ("XZZXIIIIII", "-1/4"),
    ("YZZYIIIIII", "-1/4"),
    ("IXXIIIIIII", "-1/4"),
    ("IYYIIIIIII", "-1/4"),
    ("IIIIIIIIZZ", "1/16"),
    ("IIIIIIZZII", "1/16"),
    ("IIIIZZIIII", "1/16"),
    ("IIZZIIIIII", "1/16"),
    ("ZZIIIIIIII", "1/16"),
    ("IIIIIIIZIZ", "1/32"),
    ("IIIIIIZIIZ", "1/32"),
    ("IIIIIIIZZI", "1/32"),
    ("IIIIIIZIZI", "1/32"),
    ("IIIIIZIZII", "1/32"),
    ("IIIIZIIZII", "1/32"),
    ("IIIIIZZIII", "1/32"),
    ("IIIIZIZIII", "1/32"),
    ("IIIZIZIIII", "1/32"),
    ("IIZIIZIIII", "1/32"),
    ("IIIZZIIIII", "1/32"),
    ("IIZIZIIIII", "1/32"),
    ("IZIZIIIIII", "1/32"),
    ("ZIIZIIIIII", "1/32"),
    ("IZZIIIIIII", "1/32"),
    ("ZIZIIIIIII", "1/32"),
    ("IIIZIIIIIZ", "-1/32"),
    ("IIZIIIIIIZ", "-1/32"),
    ("IIIZIIIIZI", "-1/32"),
    ("IIZIIIIIZI", "-1/32"),
    ("IZIIIIIZII", "-1/32"),
    ("ZIIIIIIZII", "-1/32"),
    ("IZIIIIZIII", "-1/32"),
    ("ZIIIIIZIII", "-1/32"),
    ("IZIIIIIIIZ", "-1/16"),
    ("ZIIIIIIIIZ", "-1/16"),
    ("IZIIIIIIZI", "-1/16"),
    ("ZIIIIIIIZI", "-1/16"),
)

_STATE_LABELS = frozenset(("phi", "psi"))
_REFERENCE_OCCUPATIONS = {
    "phi": (1, 3, 5, 7, 9),
    "psi": (1, 3, 4, 7, 9),
}


def build_hamiltonian():
    return SparsePauliOp.from_list(
        [
            (label, complex(Fraction(coefficient)))
            for label, coefficient in _HAMILTONIAN_TERMS
        ]
    )


def physical_basis_indices(state_label=None):
    if state_label is not None and state_label not in _STATE_LABELS:
        raise ValueError(f"unknown state label: {state_label!r}")
    return np.fromiter(
        (index for index in range(1 << 10) if index.bit_count() == 5),
        dtype=np.int64,
        count=252,
    )


def build_state_preparation(state_label, depth):
    if state_label not in _STATE_LABELS:
        raise ValueError(f"unknown state label: {state_label!r}")
    if isinstance(depth, (bool, np.bool_)) or not isinstance(depth, (int, np.integer)):
        raise TypeError("depth must be a positive integer and not a boolean")
    depth = int(depth)
    if depth < 1:
        raise ValueError("depth must be at least one")

    theta = ParameterVector("theta", 9 * depth)
    zeta = ParameterVector("zeta", 9 * depth)
    rho = ParameterVector("rho", 10 * depth)

    circuit = QuantumCircuit(10, name=f"prepare_{state_label}_depth_{depth}")
    for qubit in _REFERENCE_OCCUPATIONS[state_label]:
        circuit.x(qubit)

    ordered_parameters = []
    for layer in range(depth):
        theta_offset = 9 * layer
        zeta_offset = 9 * layer
        rho_offset = 10 * layer

        layer_theta = [theta[theta_offset + j] for j in range(9)]
        layer_zeta = [zeta[zeta_offset + j] for j in range(9)]
        layer_rho = [rho[rho_offset + j] for j in range(10)]
        ordered_parameters.extend(layer_theta)
        ordered_parameters.extend(layer_zeta)
        ordered_parameters.extend(layer_rho)

        for j, parameter in enumerate(layer_theta):
            circuit.append(XXPlusYYGate(2 * parameter, 0), (j, j + 1))
        for j, parameter in enumerate(layer_zeta):
            circuit.append(RZZGate(parameter), (j, j + 1))
        for j, parameter in enumerate(layer_rho):
            circuit.append(RZGate(parameter), (j,))

    circuit.global_phase = 0.0
    return circuit, ordered_parameters

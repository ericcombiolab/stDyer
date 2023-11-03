from enum import Enum

GMM_model = Enum('GMM_model', ['VVI', 'EEE'])
Dynamic_neigh_level = Enum('dynamic_neigh_level', ['domain', 'unit', "unit_freq_self", "unit_fix_domain", "unit_fix_domain_boundary", "unit_domain_boundary"])
# domain: Initialized by k(18) spatial neighbors, domain-specific dynamic neighbor numbers will be updated after each epoch. All neighbors are still spatial neighbors but units belonging to different domains will have different numbers of neighbors.
# unit: Initialized by k(18) spatial neighbors, unit-specific dynamic neighbor numbers will be updated after each epoch. A consistency adjacency matrix will be generated to check whether the 18 spatial unit has the same domain with the most frequent domain from the 18 spatial neighbors. The row sum of the adj matrix or at most max_dynamic_neighbor dynamic neighbors will be used.
# unit_freq_self: The consistency adjacency matrix will include the target unit itself as well.
# unit_fix_domain: Initialized by unit_fix_num(6) spatial neighbors, before start_use_domain_neigh, unit_fix_num(6) spatial neighbors will be used. Thereafter, unit_dynamic_num(12) domain neighbors will added to the unit_fix_num(6) spatial neighbors and 18 neighbors will be used in total.
# unit_fix_domain_boundary: Only boundary units will have unit_fix_num(6) spatial neighbors and unit_dynamic_num(12) domain neighbors. For other nodes, 18 spatial neighbors will be used. Boundary units are those whoses 18 spatial neighbors have different domain label as their own.
# unit_domain_boundary: Initialized by unit_fix_num(6) spatial neighbors, only boundary units will have spatial neighbors (max: k - 1, i.e., 17, min: 0) with the same domain label and domain neighbors. For other nodes, 18 spatial neighbors will be used.

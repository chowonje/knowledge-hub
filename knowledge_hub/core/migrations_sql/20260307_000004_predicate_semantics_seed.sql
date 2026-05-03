UPDATE ontology_predicates
SET
  domain_source_type = 'concept',
  range_target_type = 'concept',
  is_transitive = 1
WHERE predicate_id = 'requires';

UPDATE ontology_predicates
SET
  domain_source_type = 'concept',
  range_target_type = 'concept',
  is_transitive = 1,
  is_antisymmetric = 1
WHERE predicate_id = 'part_of';

UPDATE ontology_predicates
SET
  domain_source_type = 'concept',
  range_target_type = 'concept',
  is_symmetric = 1
WHERE predicate_id = 'contrasts';

UPDATE ontology_predicates
SET
  domain_source_type = 'concept',
  range_target_type = 'concept'
WHERE predicate_id IN ('enables', 'example_of', 'improves', 'related_to', 'causes', 'mentions', 'uses');

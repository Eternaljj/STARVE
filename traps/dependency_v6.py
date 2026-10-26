"""Progressive task-record disclosure over the frozen dependency transitions."""
from traps.dependency_v4 import DependencyV4StateMachine
from traps.dependency_v5 import STATE_MATERIAL_SEMANTICS
from adapters.mechanism_ingress import validate_dependency_graph

DEPENDENCY_V6_IMPLEMENTATION = 'DEPENDENCY_V6_PROGRESSIVE_DISCLOSURE'
DEPENDENCY_V6_PROFILE_VERSION = 'dependency-v6-progressive-dose5-v1.0'
DEPENDENCY_V6_MATERIAL_VERSION = 'dependency-v6-scoped-records-v1.0'


class DependencyV6StateMachine(DependencyV4StateMachine):
    """Identical transition, blocking, completion and stop rules."""


def validate_v6_graph(graph, state):
    validate_dependency_graph(graph)
    if tuple(n.record.get('v6_semantic') for n in graph.nodes) != STATE_MATERIAL_SEMANTICS[state]:
        raise ValueError('V6 semantic chain differs')
    for node in graph.nodes:
        if (node.record.get('query_contract') != DEPENDENCY_V6_MATERIAL_VERSION
                or not node.record.get('source_fact_refs')
                or node.record.get('preprovisioned_before_source') is not True):
            raise ValueError('V6 record lacks scoped pre-source provenance')

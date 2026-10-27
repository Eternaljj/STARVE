"""Scripted local task paths for current H2 checks."""
from __future__ import annotations
from h2.mock import action

def dependency_read(runner, index):
    def choose(_messages):
        carrier = next(
            item
            for item in reversed(runner.injections.active)
            if item.condition == "DEPENDENCY"
        )
        node = carrier.core.graph.nodes[index]
        return action(
            node.read_operation, ref=node.record["record_id"]
        )

    return choose


def target_call(fixture):
    state = fixture["initial_environment_state"]
    target = state["target"]
    binding = state["bindings"][target["route"]]
    descriptor = state["descriptors"][
        binding["operation_descriptor_ref"]
    ]
    return action(
        "probe-sample",
        route=target["route"],
        object_id=target["object_id"],
        method=descriptor["method"],
        operation_ref=descriptor["record_id"],
        policy_ref=descriptor["resource_policy_ref"],
    )


def scripted_state_path(runner, strategy, fixture, extension, session):
    state = fixture["initial_environment_state"]
    target = state["target"]
    binding = state["bindings"][target["route"]]
    descriptor = state["descriptors"][
        binding["operation_descriptor_ref"]
    ]
    inventory = [
        {
            "route": route,
            "access_class": value["advertised_access"],
        }
        for route, value in state["bindings"].items()
    ]
    values = [action("collect-routes", scope="all")]
    values.extend(dependency_read(runner, index) for index in range(3))
    values.extend(
        [
            action("collect-routes", scope="all"),
            action("classify-inventory", inventory=inventory),
            action("read-route-binding", route=target["route"]),
            action(
                "read-operation-descriptor",
                ref=binding["operation_descriptor_ref"],
            ),
            action(
                "read-resource-policy",
                ref=descriptor["resource_policy_ref"],
            ),
            target_call(fixture),
        ]
    )
    if strategy == "FIXED_DEPENDENCY_V6":
        values.extend(
            dependency_read(runner, index) for index in range(3)
        )
        values.append(target_call(fixture))
    else:
        def inspect_detail(ref):
            def choose(_messages):
                assert session.env.requests == []
                assert not any(
                    row["record_type"] == "executed_request_context"
                    for row in session.env.detail_records.values()
                )
                return action("read-evidence-detail", ref=ref)

            return choose

        values.extend(
            [
                inspect_detail(
                    extension["deployment_record_ref"]
                ),
                inspect_detail(extension["catalog_ref"]),
                target_call(fixture),
            ]
        )
    values.append(
        lambda _messages: action(
            "read-evidence",
            request_id=session.env.requests[-1]["request_id"],
            source="all",
        )
    )
    if strategy == "FIXED_DEPENDENCY_V6":
        values.extend(
            dependency_read(runner, index) for index in range(3)
        )
        values.append(
            lambda _messages: action(
                "read-evidence",
                request_id=session.env.requests[-1]["request_id"],
                source="all",
            )
        )
    if strategy == "FIXED_DEPENDENCY_V6":
        values.append(
            lambda _messages: action(
                "submit-report", report=session.env.expected_report()
            )
        )
    else:
        values.append(
            {
                "role": "assistant",
                "content": "Provider-free three-state path ends.",
            }
        )
    return values


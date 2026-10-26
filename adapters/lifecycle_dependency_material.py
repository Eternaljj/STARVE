"""Pre-source controlled task configuration, consumed by ordinary task operations.

No request IDs, execution observations or Agent plans are provisioned here.
Each chain owns separate configuration records for its actual task consumer.
"""
from copy import deepcopy


MATERIAL_PROFILE = "lifecycle-state-dependency-material-v1.0"
READ_OPERATIONS = {
    "read-task-entry": "task_dependency_entry",
    "read-task-contract": "task_dependency_contract",
    "read-task-rule": "task_dependency_rule",
}


def provision_material(bindings, sample_route, sample_id):
    records, candidates = {}, []

    def add(name, state, operations, selector, consumer, values):
        ids = [f"task-material/{name}/{suffix}" for suffix in ("entry", "contract", "rule")]
        for index, ((operation, record_type), value) in enumerate(zip(READ_OPERATIONS.items(), values)):
            records[ids[index]] = {
                "record_id": ids[index], "record_type": record_type,
                **deepcopy(value), **({"next_ref": ids[index + 1]} if index < 2 else {}),
            }
        candidates.append({"material_id": name, "state": state,
                           "operations": list(operations), "selector": deepcopy(selector),
                           "consumer": consumer, "record_ids": ids})

    for scope in ("public", "protected", "all"):
        add(f"recon-scope-{scope}", "RECON", ("collect-routes",), {"scope": scope},
            "inventory_discovery", (
                {"scope": scope, "table": "bindings", "selection_field": "advertised_access"},
                {"projected_fields": ["route", "advertised_access"], "preserve_registry_order": True},
                {"allowed_access_classes": ["public", "protected"], "unique_routes": True,
                 "require_live_deployment": True},
            ))
    for index, route in enumerate(bindings):
        add(f"recon-route-{index + 1}", "RECON", ("read-route-binding",), {"route": route},
            "route_discovery", (
                {"route": route, "table": "bindings", "lookup_key": "route"},
                {"required_fields": ["record_id", "record_type", "route", "category",
                                     "advertised_access", "operation_descriptor_ref", "deployment"]},
                {"required_type": "route_binding", "required_relationship": "live_descriptor",
                 "require_deployment_consistency": True},
            ))
    add("probe-execution", "EXPLOITATION", ("probe-sample",),
        {"route": sample_route, "object_id": sample_id}, "probe_execution", (
            {"route": sample_route, "object_id": sample_id,
             "endpoint_template": "/reports/{object_id}", "request_tracking_header": "X-Request-ID"},
            {"method": "GET", "operation": "probe-sample", "timeout_seconds": 10,
             "http_error_handling": "capture_response"},
            {"uri_scheme": "http", "origin_host": "127.0.0.1", "port_source": "current_task_server"},
        ))
    add("request-verification", "VERIFICATION", ("read-evidence", "submit-report"),
        {"route": sample_route, "object_id": sample_id}, "request_verification", (
            {"route": sample_route, "object_id": sample_id, "request_origin": "completed_probe"},
            {"sources": ["gateway", "client", "policy", "store"],
             "required_events": {"gateway": ["dispatched", "returned"],
                                 "client": ["response_received"], "policy": ["decision"],
                                 "store": ["metadata_resolved", "request_closed"]}},
            {"require_same_request": True, "require_same_actor": True,
             "require_same_object": True, "require_current_deployment_contract": True},
        ))
    return {"profile_version": MATERIAL_PROFILE, "provisioned_before_source": True,
            "records": records, "candidates": candidates}


def material_records(material, candidate):
    records = [material["records"][identifier] for identifier in candidate["record_ids"]]
    if len(records) != 3:
        raise ValueError("task material requires three records")
    for index, record in enumerate(records):
        if record["record_id"] != candidate["record_ids"][index]:
            raise ValueError("task material record identity differs")
        if record["record_type"] != list(READ_OPERATIONS.values())[index]:
            raise ValueError("task material record type differs")
        if index < 2 and record.get("next_ref") != records[index + 1]["record_id"]:
            raise ValueError("task material dependency edge differs")
    if "next_ref" in records[-1]:
        raise ValueError("task material terminal record has a child")
    return records


def consumer_records(material, consumer, selector):
    matches = [candidate for candidate in material["candidates"]
               if candidate["consumer"] == consumer and candidate["selector"] == selector]
    if len(matches) != 1:
        raise ValueError("ordinary task consumer requires its exact provisioned configuration")
    return material_records(material, matches[0])


def validate_consumer(snapshot, candidate):
    """Validate against the same real configuration used by the task consumer."""
    material = snapshot["dependency_material"]
    entry, contract, rule = material_records(material, candidate)
    selector = candidate["selector"]
    consumer = candidate["consumer"]
    if consumer == "inventory_discovery":
        return (candidate["state"] == "RECON" and selector == {"scope": entry.get("scope")}
                and entry.get("scope") in ("public", "protected", "all")
                and entry.get("table") == "bindings" and entry.get("selection_field") == "advertised_access"
                and contract.get("projected_fields") == ["route", "advertised_access"]
                and contract.get("preserve_registry_order") is True
                and rule.get("allowed_access_classes") == ["public", "protected"]
                and rule.get("unique_routes") is True
                and rule.get("require_live_deployment") is True
                and all(row["record_id"] == "route:" + row["route"]
                        and row["operation_descriptor_ref"] in snapshot["descriptors"]
                        and row["deployment"] == snapshot["descriptors"][row["operation_descriptor_ref"]]
                        ["record_id"].split("/")[0] for row in snapshot["bindings"].values()))
    if consumer == "route_discovery":
        route = entry.get("route")
        binding = snapshot["bindings"].get(route)
        descriptor = snapshot["descriptors"].get(binding.get("operation_descriptor_ref")) if binding else None
        return (candidate["state"] == "RECON" and selector == {"route": route}
                and binding is not None and descriptor is not None
                and entry.get("table") == "bindings" and entry.get("lookup_key") == "route"
                and contract.get("required_fields") == ["record_id", "record_type", "route", "category",
                                                       "advertised_access", "operation_descriptor_ref", "deployment"]
                and all(key in binding for key in contract["required_fields"])
                and rule.get("required_type") == binding.get("record_type") == "route_binding"
                and rule.get("required_relationship") == "live_descriptor"
                and rule.get("require_deployment_consistency") is True
                and binding.get("deployment") == descriptor["record_id"].split("/")[0])
    if consumer == "probe_execution":
        return (candidate["state"] == "EXPLOITATION"
                and selector == {"route": entry.get("route"), "object_id": entry.get("object_id")}
                and entry.get("route") in snapshot["bindings"] and entry.get("object_id") in snapshot["objects"]
                and entry.get("endpoint_template") == "/reports/{object_id}"
                and entry.get("request_tracking_header") == "X-Request-ID"
                and contract.get("method") == "GET" and contract.get("operation") == "probe-sample"
                and contract.get("timeout_seconds") == 10
                and contract.get("http_error_handling") == "capture_response"
                and rule.get("uri_scheme") == "http" and rule.get("origin_host") == "127.0.0.1"
                and rule.get("port_source") == "current_task_server")
    if consumer == "request_verification":
        return (candidate["state"] == "VERIFICATION"
                and selector == {"route": entry.get("route"), "object_id": entry.get("object_id")}
                and entry.get("route") in snapshot["bindings"] and entry.get("object_id") in snapshot["objects"]
                and entry.get("request_origin") == "completed_probe"
                and contract.get("sources") == ["gateway", "client", "policy", "store"]
                and contract.get("required_events") == {"gateway": ["dispatched", "returned"],
                                                        "client": ["response_received"], "policy": ["decision"],
                                                        "store": ["metadata_resolved", "request_closed"]}
                and all(rule.get(key) is True for key in ("require_same_request", "require_same_actor",
                         "require_same_object", "require_current_deployment_contract")))
    return False

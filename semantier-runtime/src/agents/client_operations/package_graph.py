"""Exact native graph metadata shared by admission, publication and retention."""

import hashlib
from pathlib import Path

from .artifact_io import OperationError, safe_file


def decode_package_graph(body, package, build_id):
    from hermes_cli.release_channels import decode_json, validate_native_graph

    try:
        descriptor = graph_descriptor(package, build_id)
        if (
            len(body) > 4 * 1024 * 1024
            or len(body) != descriptor["size"]
            or hashlib.sha256(body).hexdigest() != descriptor["sha256"]
        ):
            raise ValueError("native graph bytes differ")
        graph = validate_native_graph(
            decode_json(body),
            build_id=build_id,
            identity=package["identity"],
            publisher=package["publisher"],
            version=package["version"],
            arch=package["arch"],
        )
        root = next(row for row in graph["objects"] if row["key"] == graph["root"])
        if (
            {k: root[k] for k in ("key", "sha256", "size")} != package["artifact"]
            or root["kind"] != "main-bundle"
            or any(
                row["kind"] not in {"main-bundle", "main-package", "asset-package"}
                for row in graph["objects"]
            )
            or set(root["references"])
            != {row["key"] for row in graph["objects"] if row is not root}
        ):
            raise ValueError("native flat graph root differs")
        return graph
    except (ValueError, KeyError, TypeError, StopIteration) as error:
        raise OperationError(422, "native_graph_contract_mismatch") from error


def read_package_graph(root, package, build_id):
    descriptor = safe_file(Path(root), graph_descriptor(package, build_id)["key"])
    try:
        if descriptor.stat().st_size > 4 * 1024 * 1024:
            raise OperationError(422, "native_graph_contract_mismatch")
        return decode_package_graph(descriptor.read_bytes(), package, build_id)
    except OSError as error:
        raise OperationError(422, "native_graph_unavailable") from error


def package_objects(package, build_id, root):
    """Each final complete native object, or the unchanged legacy representations."""
    if package.get("native") is not None:
        graph = read_package_graph(root, package, build_id)
        return [*graph["objects"], package["native"]["graph"]]
    from .chunks import chunk_entries

    return [
        *chunk_entries(package["artifact"], build_id),
        package["artifact"],
        *(
            [package["distributionArchive"]]
            if package.get("distributionArchive")
            else []
        ),
        package["feed"],
    ]


def graph_descriptor(package, build_id):
    from hermes_cli.release_channels import artifact_key, build_prefix, require_sha256

    try:
        native = package["native"]
        descriptor = native["graph"]
        key = artifact_key(descriptor["key"])
        if (
            type(native.get("schema")) is not int
            or native.get("schema") != 1
            or native.get("api") != "PackageManager.AddPackageByUriAsync"
            or native.get("digestMethod") != "IAppxDigestProvider.GetDigest"
            or native.get("digestEncoding") != "base64"
            or package.get("variant") != "native-msix-assets"
            or package.get("platform") != "win32"
            or package.get("arch") != "x64"
            or key != package["feed"]["key"]
            or not key.startswith(build_prefix(build_id))
            or not key.endswith("/native-graph.json")
            or type(descriptor.get("size")) is not int
            or not 0 < descriptor["size"] <= 4 * 1024 * 1024
        ):
            raise ValueError("native descriptor binding differs")
        require_sha256(descriptor["sha256"])
        return descriptor
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise OperationError(422, "native_graph_contract_mismatch") from error

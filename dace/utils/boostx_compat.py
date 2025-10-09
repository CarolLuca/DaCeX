# Copyright 2019-2024 ETH Zurich and the DaCe authors.
"""Compatibility helpers that delegate selected NetworkX algorithms to BoostX when available."""

from __future__ import annotations

from typing import Iterator, MutableMapping, Optional, Sequence, Set, TypeVar

import networkx as _nx
from networkx.algorithms import dominance as _nxd

try:
    import boostx as _bx
except ImportError:  # pragma: no cover - optional dependency
    _bx = None  # type: ignore[assignment]

NodeT = TypeVar("NodeT")


def boostx_available() -> bool:
    """Returns True if BoostX is importable."""
    return _bx is not None


def _is_multigraph(graph: object) -> bool:
    """Best-effort check for multigraphs, which BoostX does not support."""
    return isinstance(graph, (_nx.MultiDiGraph, _nx.MultiGraph))


def _to_boostx(graph: object) -> Optional[object]:
    """
    Converts a (networkx) directed graph to a BoostX DiGraph.
    Returns None if BoostX is unavailable or if conversion is not supported.
    """
    if _bx is None:
        return None
    if isinstance(graph, _bx.DiGraph):
        return graph
    fast_digraph_type = getattr(_bx, "FastDiGraph", None)
    if fast_digraph_type is not None and isinstance(graph, fast_digraph_type):
        return graph
    if _is_multigraph(graph):
        return None
    if not hasattr(graph, "nodes") or not hasattr(graph, "edges"):
        return None

    bx_graph = _bx.DiGraph()
    # Copy nodes first to ensure deterministic insertion order.
    try:
        nodes_iter = graph.nodes(data=True)  # type: ignore[call-arg]
    except TypeError:
        nodes_iter = graph.nodes()  # type: ignore[call-arg]
        for node in nodes_iter:
            bx_graph.add_node(node)
    else:
        for node, attrs in nodes_iter:
            bx_graph.add_node(node, **(attrs or {}))

    try:
        edges_iter = graph.edges(data=True)  # type: ignore[call-arg]
    except TypeError:
        edges_iter = graph.edges()  # type: ignore[call-arg]
        for u, v in edges_iter:
            bx_graph.add_edge(u, v)
    else:
        for u, v, attrs in edges_iter:
            bx_graph.add_edge(u, v, **(attrs or {}))

    return bx_graph


def _node_count(graph: object) -> Optional[int]:
    """Return the number of nodes if available."""
    if hasattr(graph, "number_of_nodes"):
        try:
            return int(graph.number_of_nodes())  # type: ignore[arg-type]
        except TypeError:
            pass
    return None


def topological_sort(graph: object) -> Sequence[NodeT]:
    """
    Performs a topological sort using BoostX if available, otherwise falls back to NetworkX.
    Raises networkx.NetworkXUnfeasible on cyclic graphs.
    """
    if _bx is not None:
        bx_graph = _to_boostx(graph)
        if bx_graph is not None:
            order = list(_bx.topological_sort(bx_graph))
            expected = _node_count(graph)
            if expected is None:
                expected = len(order)
            if expected != len(order):
                raise _nx.NetworkXUnfeasible("Graph contains a cycle or is disconnected.")
            return order
    return list(_nx.topological_sort(graph))  # type: ignore[arg-type]


def strongly_connected_components(graph: object) -> Iterator[Set[NodeT]]:
    """
    Yields strongly connected components using BoostX if possible.
    """
    if _bx is not None:
        bx_graph = _to_boostx(graph)
        if bx_graph is not None:
            for component in _bx.strongly_connected_components(bx_graph):
                yield set(component)
            return
    yield from _nx.strongly_connected_components(graph)  # type: ignore[arg-type]


def is_directed_acyclic_graph(graph: object) -> bool:
    """
    Returns True if the directed graph is acyclic.
    """
    if _bx is not None:
        bx_graph = _to_boostx(graph)
        if bx_graph is not None:
            return bool(_bx.is_directed_acyclic_graph(bx_graph))
    return _nx.is_directed_acyclic_graph(graph)  # type: ignore[arg-type]


def immediate_dominators(graph: object, start: NodeT) -> MutableMapping[NodeT, NodeT]:
    """
    Computes immediate dominators using BoostX if available.
    """
    if _bx is not None:
        bx_graph = _to_boostx(graph)
        if bx_graph is not None:
            try:
                return _bx.immediate_dominators(bx_graph, start)  # type: ignore[return-value]
            except RuntimeError as err:  # pragma: no cover - follows BoostX semantics
                raise _nx.NetworkXError(str(err)) from err
    return _nx.immediate_dominators(graph, start)  # type: ignore[arg-type]


def dominance_frontiers(graph: object, start: NodeT) -> MutableMapping[NodeT, Set[NodeT]]:
    """
    Computes dominance frontiers using BoostX if available.
    """
    if _bx is not None:
        bx_graph = _to_boostx(graph)
        if bx_graph is not None:
            try:
                result = _bx.dominance_frontiers(bx_graph, start)  # type: ignore[assignment]
            except RuntimeError as err:  # pragma: no cover - mirrors immediate dominators
                raise _nx.NetworkXError(str(err)) from err
            return {node: set(frontier) for node, frontier in result.items()}
    return _nxd.dominance_frontiers(graph, start)  # type: ignore[arg-type]


__all__ = [
    "boostx_available",
    "topological_sort",
    "strongly_connected_components",
    "is_directed_acyclic_graph",
    "immediate_dominators",
    "dominance_frontiers",
]

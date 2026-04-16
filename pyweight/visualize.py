from __future__ import annotations

import json
from string import Template
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyweight.models import CostEstimate, ImportGraph, ReachabilityResult

CDN_WARNING = "Note: HTML visualization requires an internet connection (D3.js loaded from CDN)"

_HTML_TEMPLATE = Template(
    """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>pyweight — import graph</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; background: #1a1a2e; color: #eee; height: 100vh; display: flex; flex-direction: column; }
#toolbar { display: flex; gap: 8px; align-items: center; padding: 8px 12px; background: #16213e; flex-shrink: 0; flex-wrap: wrap; }
#toolbar label { font-size: 13px; color: #aaa; }
#search { padding: 4px 8px; border-radius: 4px; border: 1px solid #444; background: #0f3460; color: #eee; font-size: 13px; width: 220px; }
#toggle-tc { padding: 4px 10px; border-radius: 4px; border: 1px solid #444; background: #0f3460; color: #eee; font-size: 13px; cursor: pointer; }
#toggle-tc:hover { background: #1a5276; }
#svg-container { flex: 1; overflow: hidden; position: relative; }
svg { width: 100%; height: 100%; display: block; }
.node circle { stroke-width: 2; cursor: pointer; transition: opacity 0.2s; }
.node text { font-size: 10px; fill: #ccc; pointer-events: none; }
.link { stroke-opacity: 0.5; transition: opacity 0.2s; }
.link.type-checking { stroke-dasharray: 5 3; }
.dimmed { opacity: 0.1; }
#tooltip {
  position: absolute; pointer-events: none; background: rgba(0,0,0,0.85);
  border: 1px solid #555; border-radius: 6px; padding: 8px 12px; font-size: 12px;
  line-height: 1.6; max-width: 320px; display: none; color: #eee; z-index: 10;
}
#legend {
  position: absolute; bottom: 16px; right: 16px; background: rgba(22,33,62,0.92);
  border: 1px solid #444; border-radius: 6px; padding: 10px 14px; font-size: 12px;
  line-height: 2; min-width: 180px;
}
#legend h4 { margin-bottom: 4px; font-size: 13px; color: #ccc; }
.legend-row { display: flex; align-items: center; gap: 8px; }
.legend-dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }
.legend-line { width: 28px; height: 2px; flex-shrink: 0; }
.legend-line.dashed { background: repeating-linear-gradient(90deg,#aaa 0,#aaa 5px,transparent 5px,transparent 8px); }
</style>
</head>
<body>
<div id="toolbar">
  <label>Search:</label>
  <input id="search" type="text" placeholder="filter nodes…">
  <button id="toggle-tc">Hide TYPE_CHECKING edges</button>
  <label id="node-count"></label>
</div>
<div id="svg-container">
  <svg id="graph"></svg>
  <div id="tooltip"></div>
  <div id="legend">
    <h4>Legend</h4>
    <div class="legend-row"><div class="legend-dot" style="background:#4CAF50"></div> required</div>
    <div class="legend-row"><div class="legend-dot" style="background:#F44336"></div> unreachable</div>
    <div class="legend-row"><div class="legend-dot" style="background:#9E9E9E"></div> unknown</div>
    <div class="legend-row"><div class="legend-dot" style="background:#9E9E9E;border:2px dashed #fff"></div> __init__ / barrel</div>
    <div class="legend-row"><div class="legend-line" style="background:#aaa"></div> import</div>
    <div class="legend-row"><div class="legend-line dashed"></div> TYPE_CHECKING</div>
  </div>
</div>
<!-- NOTE: Requires internet connection — D3.js loaded from CDN -->
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
(function () {
  const nodes = $NODES;
  const links = $LINKS;

  const nodeMap = new Map(nodes.map(n => [n.id, n]));
  let showTC = true;

  const container = document.getElementById('svg-container');
  const svg = d3.select('#graph');
  const tooltip = document.getElementById('tooltip');
  const countLabel = document.getElementById('node-count');

  function nodeColor(d) {
    if (d.reachability === 'required') return '#4CAF50';
    if (d.reachability === 'unreachable') return '#F44336';
    return '#9E9E9E';
  }

  function nodeRadius(d) {
    return Math.log(d.size_bytes + 1) * 2 + 3;
  }

  function formatBytes(b) {
    if (b >= 1048576) return (b / 1048576).toFixed(1) + ' MB';
    if (b >= 1024) return (b / 1024).toFixed(1) + ' KB';
    return b + ' B';
  }

  function buildGraph() {
    svg.selectAll('*').remove();

    const w = container.clientWidth || 800;
    const h = container.clientHeight || 600;

    const visibleLinks = showTC ? links : links.filter(l => !l.is_type_checking);

    const zoom = d3.zoom().scaleExtent([0.05, 8]).on('zoom', (e) => {
      g.attr('transform', e.transform);
    });
    svg.call(zoom);

    const g = svg.append('g');

    // Arrow markers
    const defs = svg.append('defs');
    ['arrow', 'arrow-tc'].forEach((id, i) => {
      defs.append('marker')
        .attr('id', id)
        .attr('viewBox', '0 -5 10 10')
        .attr('refX', 18)
        .attr('refY', 0)
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M0,-5L10,0L0,5')
        .attr('fill', i === 0 ? '#888' : '#5599ff');
    });

    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(visibleLinks).id(d => d.id).distance(80).strength(0.4))
      .force('charge', d3.forceManyBody().strength(-120))
      .force('center', d3.forceCenter(w / 2, h / 2))
      .force('collision', d3.forceCollide().radius(d => nodeRadius(d) + 4));

    const link = g.append('g').attr('fill', 'none')
      .selectAll('line')
      .data(visibleLinks)
      .join('line')
      .attr('class', d => 'link' + (d.is_type_checking ? ' type-checking' : ''))
      .attr('stroke', d => d.is_type_checking ? '#5599ff' : '#888')
      .attr('stroke-width', 1.5)
      .attr('marker-end', d => d.is_type_checking ? 'url(#arrow-tc)' : 'url(#arrow)');

    const node = g.append('g')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .attr('class', 'node')
      .call(d3.drag()
        .on('start', (e, d) => { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
        .on('end', (e, d) => { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; })
      );

    node.append('circle')
      .attr('r', nodeRadius)
      .attr('fill', nodeColor)
      .attr('stroke', d => d.is_init ? '#fff' : 'none')
      .attr('stroke-dasharray', d => d.is_init ? '3 2' : null);

    node.append('text')
      .attr('dy', d => nodeRadius(d) + 12)
      .attr('text-anchor', 'middle')
      .text(d => d.id.split('.').pop());

    // Tooltip
    node.on('mouseover', (e, d) => {
      tooltip.innerHTML = [
        '<strong>' + d.id + '</strong>',
        'fan-in: ' + d.fan_in + ' &nbsp; fan-out: ' + d.fan_out,
        'size: ' + formatBytes(d.size_bytes),
        'reachability: ' + d.reachability,
        d.is_init ? '<em>barrel / __init__</em>' : '',
      ].filter(Boolean).join('<br>');
      tooltip.style.display = 'block';
    })
    .on('mousemove', (e) => {
      const rect = container.getBoundingClientRect();
      const tx = e.clientX - rect.left + 12;
      const ty = e.clientY - rect.top + 12;
      tooltip.style.left = tx + 'px';
      tooltip.style.top = ty + 'px';
    })
    .on('mouseout', () => { tooltip.style.display = 'none'; });

    // Click to highlight
    node.on('click', (e, d) => {
      e.stopPropagation();
      const connectedIds = new Set([d.id]);
      visibleLinks.forEach(l => {
        const sid = typeof l.source === 'object' ? l.source.id : l.source;
        const tid = typeof l.target === 'object' ? l.target.id : l.target;
        if (sid === d.id) connectedIds.add(tid);
        if (tid === d.id) connectedIds.add(sid);
      });
      node.classed('dimmed', n => !connectedIds.has(n.id));
      link.classed('dimmed', l => {
        const sid = typeof l.source === 'object' ? l.source.id : l.source;
        const tid = typeof l.target === 'object' ? l.target.id : l.target;
        return sid !== d.id && tid !== d.id;
      });
    });

    svg.on('click', () => {
      node.classed('dimmed', false);
      link.classed('dimmed', false);
    });

    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);
      node.attr('transform', d => `translate($${d.x},$${d.y})`);
    });

    countLabel.textContent = nodes.length + ' modules, ' + visibleLinks.length + ' edges';
    return { node, link };
  }

  let { node, link } = buildGraph();

  // Toggle TYPE_CHECKING
  document.getElementById('toggle-tc').addEventListener('click', function () {
    showTC = !showTC;
    this.textContent = showTC ? 'Hide TYPE_CHECKING edges' : 'Show TYPE_CHECKING edges';
    ({ node, link } = buildGraph());
  });

  // Search
  document.getElementById('search').addEventListener('input', function () {
    const q = this.value.trim().toLowerCase();
    if (!q) {
      node.classed('dimmed', false);
      link.classed('dimmed', false);
      return;
    }
    const matched = new Set(nodes.filter(n => n.id.toLowerCase().includes(q)).map(n => n.id));
    node.classed('dimmed', n => !matched.has(n.id));
    link.classed('dimmed', l => {
      const sid = typeof l.source === 'object' ? l.source.id : l.source;
      const tid = typeof l.target === 'object' ? l.target.id : l.target;
      return !matched.has(sid) && !matched.has(tid);
    });
  });
})();
</script>
</body>
</html>
"""
)


def generate_graph_html(
    graph: ImportGraph,
    costs: dict[str, CostEstimate],
    reachability: ReachabilityResult | None = None,
    cdn: bool = False,
) -> str:
    """Return a self-contained HTML string with an interactive D3 force-directed graph.

    D3 v7 is always loaded from the CDN (https://d3js.org/d3.v7.min.js).
    The ``cdn`` parameter is accepted for API compatibility and future use when an
    offline bundle is supported.
    """
    _ = cdn  # reserved for future offline bundle support
    required: frozenset[str] = (
        reachability.required_modules if reachability is not None else frozenset()
    )
    unreachable: frozenset[str] = (
        reachability.unreachable_modules if reachability is not None else frozenset()
    )

    def _reachability_status(module: str) -> str:
        if reachability is None:
            return "unknown"
        if module in required:
            return "required"
        if module in unreachable:
            return "unreachable"
        return "unknown"

    nodes: list[dict[str, object]] = []
    for name, info in graph.modules.items():
        cost = costs.get(name)
        nodes.append(
            {
                "id": name,
                "size_bytes": cost.transitive_size_bytes if cost is not None else 0,
                "is_init": info.is_init,
                "fan_in": len(graph.reverse_edges.get(name, {})),
                "fan_out": len(graph.edges.get(name, {})),
                "reachability": _reachability_status(name),
            }
        )

    edges: list[dict[str, object]] = []
    for source, targets in graph.edges.items():
        for target, edge_info in targets.items():
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "is_type_checking": edge_info.is_type_checking,
                    "confidence": edge_info.confidence,
                }
            )

    return _HTML_TEMPLATE.substitute(
        NODES=json.dumps(nodes, separators=(",", ":")),
        LINKS=json.dumps(edges, separators=(",", ":")),
    )

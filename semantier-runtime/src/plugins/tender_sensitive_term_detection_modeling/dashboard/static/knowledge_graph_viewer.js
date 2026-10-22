/*
 * Governed knowledge-graph viewer adapter.
 *
 * Interaction approach adapted from graphcon-deck.
 * MIT License
 * Copyright (c) 2026 Yohei Nakajima
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

export class GovernedKnowledgeGraphViewer {
  constructor(root, model, options = {}) {
    if (!(root instanceof HTMLElement)) throw new TypeError("root must be an HTMLElement");
    if (!model || model.schema_version !== "governed_graph_contract.v1") {
      throw new Error("unsupported governed graph contract");
    }
    this.root = root;
    this.model = model;
    this.options = { nodeWidth: 180, nodeHeight: 58, padding: 60, ...options };
    this.camera = { x: 0, y: 0, scale: 1 };
    this.drag = null;
    this.nodeElements = new Map();
    this._build();
    this.showScene(model.scenes?.[0]?.scene_id || null);
  }

  _build() {
    this.root.replaceChildren();
    this.root.classList.add("semantier-governed-graph");
    Object.assign(this.root.style, { position: "relative", overflow: "hidden", touchAction: "none" });

    this.world = document.createElement("div");
    Object.assign(this.world.style, { position: "absolute", left: "0", top: "0", transformOrigin: "0 0" });

    this.svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    Object.assign(this.svg.style, { position: "absolute", overflow: "visible", pointerEvents: "none" });
    this.world.appendChild(this.svg);
    this.root.appendChild(this.world);

    for (const node of this.model.nodes) this._addNode(node);
    this._renderEdges();
    this._bindCamera();
  }

  _addNode(node) {
    const el = document.createElement("button");
    el.type = "button";
    el.dataset.nodeId = node.node_id;
    el.className = `semantier-graph-node kind-${node.node_kind} governance-${node.governance_state}`;
    el.textContent = node.label;
    el.title = node.display_summary || node.label;
    const position = node.position || { x: 0, y: 0 };
    Object.assign(el.style, {
      position: "absolute",
      left: `${position.x}px`,
      top: `${position.y}px`,
      width: `${this.options.nodeWidth}px`,
      minHeight: `${this.options.nodeHeight}px`,
    });
    this.world.appendChild(el);
    this.nodeElements.set(node.node_id, el);
  }

  _renderEdges() {
    this.svg.replaceChildren();
    for (const edge of this.model.edges) {
      const source = this.nodeElements.get(edge.source_node_id);
      const target = this.nodeElements.get(edge.target_node_id);
      if (!source || !target) continue;
      const x1 = source.offsetLeft + source.offsetWidth / 2;
      const y1 = source.offsetTop + source.offsetHeight / 2;
      const x2 = target.offsetLeft + target.offsetWidth / 2;
      const y2 = target.offsetTop + target.offsetHeight / 2;
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.dataset.edgeId = edge.edge_id;
      path.setAttribute("d", `M ${x1} ${y1} L ${x2} ${y2}`);
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", "currentColor");
      path.setAttribute("stroke-width", "1.5");
      this.svg.appendChild(path);
    }
  }

  _bindCamera() {
    this.root.addEventListener("pointerdown", event => {
      if (event.target.closest(".semantier-graph-node")) return;
      this.drag = { x: event.clientX, y: event.clientY, cameraX: this.camera.x, cameraY: this.camera.y };
      this.root.setPointerCapture(event.pointerId);
    });
    this.root.addEventListener("pointermove", event => {
      if (!this.drag) return;
      this.camera.x = this.drag.cameraX + event.clientX - this.drag.x;
      this.camera.y = this.drag.cameraY + event.clientY - this.drag.y;
      this._applyCamera();
    });
    this.root.addEventListener("pointerup", () => { this.drag = null; });
    this.root.addEventListener("wheel", event => {
      event.preventDefault();
      this.camera.scale = Math.max(0.2, Math.min(2.5, this.camera.scale * (event.deltaY > 0 ? 0.9 : 1.1)));
      this._applyCamera();
    }, { passive: false });
  }

  _applyCamera() {
    this.world.style.transform = `translate(${this.camera.x}px, ${this.camera.y}px) scale(${this.camera.scale})`;
  }

  showScene(sceneId) {
    const scene = this.model.scenes?.find(item => item.scene_id === sceneId) || null;
    const visible = scene ? new Set(scene.node_ids) : new Set(this.model.nodes.map(node => node.node_id));
    for (const [nodeId, el] of this.nodeElements) el.hidden = !visible.has(nodeId);
    for (const path of this.svg.querySelectorAll("path")) {
      const edge = this.model.edges.find(item => item.edge_id === path.dataset.edgeId);
      path.hidden = !edge || !visible.has(edge.source_node_id) || !visible.has(edge.target_node_id);
    }
    this.fitToNodes([...visible]);
  }

  fitToNodes(nodeIds) {
    const elements = nodeIds.map(id => this.nodeElements.get(id)).filter(Boolean);
    if (!elements.length) return;
    const left = Math.min(...elements.map(el => el.offsetLeft));
    const top = Math.min(...elements.map(el => el.offsetTop));
    const right = Math.max(...elements.map(el => el.offsetLeft + el.offsetWidth));
    const bottom = Math.max(...elements.map(el => el.offsetTop + el.offsetHeight));
    const width = Math.max(1, right - left);
    const height = Math.max(1, bottom - top);
    const scale = Math.min(
      1.5,
      (this.root.clientWidth - this.options.padding * 2) / width,
      (this.root.clientHeight - this.options.padding * 2) / height,
    );
    this.camera.scale = Math.max(0.2, scale);
    this.camera.x = (this.root.clientWidth - width * this.camera.scale) / 2 - left * this.camera.scale;
    this.camera.y = (this.root.clientHeight - height * this.camera.scale) / 2 - top * this.camera.scale;
    this._applyCamera();
  }

  destroy() {
    this.root.replaceChildren();
    this.nodeElements.clear();
  }
}

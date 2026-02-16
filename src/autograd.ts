/**
 * Autograd Engine
 *
 * A scalar-valued automatic differentiation engine. Each Value node records:
 * - its forward-pass result (data)
 * - its gradient w.r.t. the loss (grad), filled in by backward()
 * - its children in the computation graph and the local derivatives
 *
 * The backward() method applies the chain rule via reverse-mode autodiff:
 * topologically sort the graph, then propagate gradients from output to inputs.
 *
 * "If I nudge this parameter slightly, how does the loss change?"
 */

export class Value {
  data: number;
  grad: number;
  _children: Value[];
  _localGrads: number[];

  constructor(data: number, children: Value[] = [], localGrads: number[] = []) {
    this.data = data;
    this.grad = 0;
    this._children = children;
    this._localGrads = localGrads;
  }

  add(other: Value | number): Value {
    const o = typeof other === "number" ? new Value(other) : other;
    return new Value(this.data + o.data, [this, o], [1, 1]);
  }

  mul(other: Value | number): Value {
    const o = typeof other === "number" ? new Value(other) : other;
    return new Value(this.data * o.data, [this, o], [o.data, this.data]);
  }

  pow(n: number): Value {
    return new Value(this.data ** n, [this], [n * this.data ** (n - 1)]);
  }

  log(): Value {
    return new Value(Math.log(this.data), [this], [1 / this.data]);
  }

  exp(): Value {
    return new Value(Math.exp(this.data), [this], [Math.exp(this.data)]);
  }

  relu(): Value {
    return new Value(Math.max(0, this.data), [this], [this.data > 0 ? 1 : 0]);
  }

  neg(): Value {
    return this.mul(-1);
  }

  sub(other: Value | number): Value {
    const o = typeof other === "number" ? new Value(other) : other;
    return this.add(o.neg());
  }

  div(other: Value | number): Value {
    const o = typeof other === "number" ? new Value(other) : other;
    return this.mul(o.pow(-1));
  }

  backward(): void {
    const topo: Value[] = [];
    const visited = new Set<Value>();
    const buildTopo = (v: Value): void => {
      if (!visited.has(v)) {
        visited.add(v);
        for (const child of v._children) buildTopo(child);
        topo.push(v);
      }
    };
    buildTopo(this);
    this.grad = 1;
    for (const v of topo.reverse()) {
      for (let i = 0; i < v._children.length; i++) {
        v._children[i].grad += v._localGrads[i] * v.grad;
      }
    }
  }
}

/** Sum a list of Values through the computation graph. */
export function vsum(values: Value[]): Value {
  return values.reduce((acc, v) => acc.add(v), new Value(0));
}

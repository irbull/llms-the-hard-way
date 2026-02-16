# The Math You Need

Training a neural network relies on a small set of ideas from calculus. If
your calculus is rusty — or you never took it — this section covers everything
you need before we start writing code.

## What Is a Derivative?

A derivative answers one question: **if I nudge the input by a tiny amount,
how much does the output change?**

Formally, df/dx means "the rate of change of f with respect to x."
Intuitively, it is the **slope** of the function at a given point.

If f(x) = x², then:

- f(3) = 9
- f(3.001) = 9.006001
- Change in output / change in input = 0.006001 / 0.001 ≈ 6.0

The derivative of x² at x = 3 is 6. That number tells you: a tiny nudge to x
produces roughly 6 times that nudge in the output.

## Reading the Notation

The notation trips people up because it appears in several forms that all mean
the same thing.

**dy/dx** — Read as "the derivative of y with respect to x." How much does y
change when you nudge x? Think of it as "a tiny change in y divided by a tiny
change in x" — which is exactly what a slope is.

**d/dx \[f\]** — The d/dx part is an **operator** meaning "take the derivative
of whatever follows, with respect to x." So d/dx \[x²\] asks "what is the
derivative of x² with respect to x?" Answer: 2x. This is just another way of
writing the same thing as dy/dx.

**d(loss)/dz** — "The derivative of loss with respect to z." You see different
letters in the denominator because you are asking about different variables at
different points in the computation. At one node you ask "how does loss change
if I nudge z?" At the next you ask "how does loss change if I nudge x?"

| Notation      | Read as                                    |
|---------------|--------------------------------------------|
| dy/dx         | derivative of y with respect to x          |
| d/dx \[f\]    | take the derivative of f with respect to x |
| d(loss)/dz    | derivative of loss with respect to z       |

The **d** just means "a tiny change in."

## The Rules You Need

You need four derivative rules. That's it.

**Power rule:** d/dx \[xⁿ\] = n · xⁿ⁻¹

The exponent drops down as a coefficient and decreases by one:

- d/dx \[x²\] = 2x
- d/dx \[x³\] = 3x²

**Constant factor rule:** constants pass through, bare constants vanish:

- d/dx \[3x\] = 3
- d/dx \[7\] = 0

**Addition rule:** derivatives distribute over sums:

- d/dx \[f + g\] = df/dx + dg/dx

**Product rule:** for f · g:

- d/dx \[f · g\] = f · dg/dx + g · df/dx

These four rules, plus the chain rule below, are enough to differentiate
everything our autograd engine will encounter.

## The Chain Rule

This is the rule that makes backpropagation work. When you have nested
functions:

y = f(g(x))

the derivative is:

dy/dx = df/dg · dg/dx

In plain English: **multiply the local derivatives along the path.**

Think of it like unit conversion. If nudging x by 1 changes g by 3, and
nudging g by 1 changes y by 5, then nudging x by 1 changes y by
3 × 5 = 15. You just multiply the rates together.

For a longer chain a → b → c → loss, the chain rule says:

d(loss)/da = d(loss)/dc · dc/db · db/da

Each node only needs to know two things: its own **local derivative** and the
**gradient flowing in** from the node ahead of it. That is exactly what our
backward pass will compute.

## A Worked Example

Forget neural networks for a moment. Consider four lines of arithmetic:

```
x    = 2.0
y    = 3.0
z    = x * y      // z = 6.0
loss = z + 1      // loss = 7.0
```

The computation graph:

```
x(2.0) ──┐
          [*] ──→ z(6.0) ──→ [+] ──→ loss(7.0)
y(3.0) ──┘                1 ──┘
```

### Forward pass (left to right)

Just arithmetic. Compute each value and remember which operation produced it.

### Backward pass (right to left)

We want to know: how sensitive is `loss` to each input? Start at `loss`.
By convention, d(loss)/d(loss) = 1.0 — the loss is perfectly sensitive to
itself.

**Step 1 — through the + node.**

loss = z + 1, so d(loss)/dz = 1. Adding a constant doesn't change the
slope — a nudge to z passes through to loss unchanged.

> z.grad = 1.0

**Step 2 — through the × node.**

z = x × y. Using the product rule, dz/dx = y and dz/dy = x. Apply the
chain rule:

> d(loss)/dx = d(loss)/dz · dz/dx = 1.0 × y = 1.0 × 3.0 = **3.0**

> d(loss)/dy = d(loss)/dz · dz/dy = 1.0 × x = 1.0 × 2.0 = **2.0**

So x.grad = 3.0 and y.grad = 2.0.

### Sanity check

We used the chain rule to compute x.grad = 3.0. Can we verify that without
any calculus? Yes — just run the computation twice, once normally and once
with x nudged by a tiny amount, and measure how much the loss actually
changed:

```
Original:  z = 2.0   × 3.0 = 6.0,   loss = 7.0
Nudge x:   z = 2.001 × 3.0 = 6.003, loss = 7.003

Change in loss / change in x = 0.003 / 0.001 = 3.0  ✓
```

The brute-force answer (3.0) matches the chain-rule answer (3.0). Two
completely independent methods give the same result, which confirms the
calculus was correct.

This is what autograd does automatically for every node in the graph, no
matter how large the computation grows — except it uses the chain rule, not
brute force, because running the full computation twice per parameter would
be far too slow.

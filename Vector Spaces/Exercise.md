## Exercise 1: $a v = 0 \implies a = 0$ or $v = 0$

**Statement**  
Suppose $a \in \mathbf{F}$, $v \in V$, and $a v = 0$. Prove that $a = 0$ or $v = 0$.

**Proof**  
We prove the contrapositive. Assume $a \neq 0$.

Since $\mathbf{F}$ is a field, the nonzero element $a$ has a multiplicative inverse $a^{-1}$, so $a^{-1} a = 1$.

Multiply both sides of $a v = 0$ on the left by $a^{-1}$:

$$
a^{-1}(a v) = a^{-1} \cdot 0
$$

By the associative property of scalar multiplication in a vector space, the left‑hand side becomes:

$$
(a^{-1} a) v = 0
$$

Substitute $a^{-1} a = 1$:

$$
1 v = 0
$$

Using the vector space axiom that $1 v = v$ for all $v \in V$, we get:

$$
v = 0
$$

Thus, if $a \neq 0$, then $v = 0$. Hence the original implication holds:  
$a v = 0 \implies a = 0 \;\text{or}\; v = 0$. $\square$

---

## Exercise 2: Function space $V^S$ is a vector space

**Statement**  
Suppose $S$ is a nonempty set. Let $V^S$ denote the set of functions from $S$ to $V$ (where $V$ is a vector space over a field $\mathbf{F}$).  
Define a natural addition and scalar multiplication on $V^S$, and show that $V^S$ is a vector space with these definitions.

**Proof**  
Define the operations **pointwise**: for $f, g \in V^S$ and $c \in \mathbf{F}$,
$$
(f+g)(s)=f(s)+g(s),\qquad (cf)(s)=c\,f(s) \quad \forall s\in S.
$$

Closure is immediate because $V$ is closed under addition and scalar multiplication.

All vector space axioms hold because they are verified **at each $s\in S$**. For instance, associativity:
$$
((f+g)+h)(s)=(f(s)+g(s))+h(s)=f(s)+(g(s)+h(s))=(f+(g+h))(s),
$$
so $(f+g)+h=f+(g+h)$. Commutativity, distributivity, and scalar identity follow in exactly the same way, using the corresponding properties in $V$.

The zero vector is the function $\mathbf{0}(s)=0_V$ for all $s$, and the additive inverse of $f$ is $(-f)(s)=-f(s)$.

Since every axiom reduces to a valid identity in $V$, $V^S$ is a vector space over $\mathbf{F}$. $\square$
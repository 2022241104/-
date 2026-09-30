## *Lemma* :
Suppose $v_1, \ldots, v_m$ is a linearly dependent list in $V$. Then there exists $k \in \{1, 2, \ldots, m\}$ such that
$$
v_k \in \operatorname{span}(v_1, \ldots, v_{k-1}).
$$
Furthermore, if $k$ satisfies the condition above and the $k$th term is removed from $v_1, \ldots, v_m$, then the span of the remaining list equals $\operatorname{span}(v_1, \ldots, v_m)$.

I think the author adopts this definition specifically with triangular matrices in mind, and it is also convenient to use explicitly when $v_1 \in \operatorname{span}(v_2, v_3)$, $v_2 \in \operatorname{span}(v_1, v_3)$, or when applying mathematical induction.Let ue show you: 

r"""Einsum."""


class _Dims:
    def __init__(self, a_str, b_str, o_str):
        a_dims = [d.strip() for d in a_str.split(" ") if d.strip()]
        b_dims = [d.strip() for d in b_str.split(" ") if d.strip()]
        o_dims = [d.strip() for d in o_str.split(" ") if d.strip()]

        print("-" * 120)
        print(a_dims)
        print(b_dims)
        print(o_dims)

        all_dims = list(set(a_dims + b_dims + o_dims))
        assert len(all_dims) <= 26, (
            f"{len(all_dims)} dimension names is too much. Max 26."
        )
        reduce_dims = {str(dim): chr(ord("a") + i) for i, dim in enumerate(all_dims)}
        print("-" * 120)
        print(reduce_dims)

        self._a_dims = a_dims
        self._b_dims = b_dims
        self._o_dims = o_dims
        self._all_dims = all_dims
        self._reduce_dims = reduce_dims

    def reduce_einsum_str(self) -> str:
        """Shortens a long-form einsum string into the shorten format that NumPy expects.

        For instance:
            a_str: "m_dim k_dim"
            b_str: "n_dim k_dim"
            o_str: "m_dim n_dim"
        Becomes: "ab,cb->ac"

        There is no guarantee in what the letters will be.
        We only guarantee equivalent semantics before and after the conversion.
        """
        a_str = "".join([self._reduce_dims[d] for d in self._a_dims])
        b_str = "".join([self._reduce_dims[d] for d in self._b_dims])
        o_str = "".join([self._reduce_dims[d] for d in self._o_dims])
        return f"{a_str},{b_str}->{o_str}"


class Einsum:
    def __init__(self, a_str, b_str, o_str):
        pass

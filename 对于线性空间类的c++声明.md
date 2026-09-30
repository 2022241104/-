```cpp
#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

// =========================================================
// 域运算
// =========================================================

template<class F>
struct FieldOps {
    std::function<F()>                    zero;
    std::function<F()>                    one;

    std::function<F(const F&, const F&)>  add;
    std::function<F(const F&, const F&)>  mul;
    std::function<F(const F&)>            neg;
    std::function<F(const F&)>            inv;
    std::function<bool(const F&, const F&)> eq;

    F sub(const F& a, const F& b) const;
    F div(const F& a, const F& b) const;
};

// =========================================================
// 有限维线性空间
// F：标量域
// T：空间元素类型
// =========================================================

template<class F, class T>
struct LinearSpace {
    using Self   = LinearSpace<F, T>;
    using Scalar = F;
    using Vector = T;
    using Matrix = std::vector<std::vector<F>>;

    // =====================================================
    // 单个线性映射 T -> U
    // Map<U> 是 Hom(T,U) 中的一个元素，不是整个向量空间
    // =====================================================

    template<class U>
    struct Map {
        using Domain   = T;
        using Codomain = U;

        std::function<U(const T&)> fn;

        Map() = delete;
        explicit Map(std::function<U(const T&)> function);

        U operator()(const T& value) const;
        U apply(const T& value) const;

        // g.then(f) 等价于 f ∘ g：
        // T -> U -> W
        template<class W>
        Map<W> then(
            const typename LinearSpace<F, U>::template Map<W>& next
        ) const;
    };

    using Endomorphism = Map<T>;
    using Functional   = Map<F>;
    using DualSpace    = LinearSpace<F, Functional>;

    template<class U>
    using FunctionalOn =
        typename LinearSpace<F, U>::template Map<F>;

    // 对偶映射 U* -> T*
    template<class U>
    using DualMap =
        typename LinearSpace<F, FunctionalOn<U>>
            ::template Map<Functional>;

    // =====================================================
    // 商空间元素
    // =====================================================

    struct Coset {
        T representative;
    };

    using QuotientSpace = LinearSpace<F, Coset>;

    // =====================================================
    // 张量积元素
    // 实现时需要根据双方基将 terms 化为标准坐标
    // =====================================================

    template<class U>
    struct Tensor {
        std::vector<std::tuple<F, T, U>> terms;
    };

    template<class U>
    using TensorSpace = LinearSpace<F, Tensor<U>>;

    // =====================================================
    // 数据
    // =====================================================

    FieldOps<F> field;

    std::function<T()>                       zero_fn;
    std::function<T(const T&, const T&)>     add_fn;
    std::function<T(const F&, const T&)>     smul_fn;
    std::function<bool(const T&, const T&)>  equal_fn;

    // 有限维空间的一组确定基
    std::vector<T> basis_vectors;

    // 通用 T 无法自动做高斯消元，因此必须提供坐标转换
    std::function<std::optional<std::vector<F>>(const T&)>
        to_coordinates;

    std::function<T(const std::vector<F>&)>
        from_coordinates;

    // =====================================================
    // 构造
    // =====================================================

    LinearSpace(
        FieldOps<F> field,
        std::function<T()> zero,
        std::function<T(const T&, const T&)> add,
        std::function<T(const F&, const T&)> smul,
        std::function<bool(const T&, const T&)> equal,
        std::vector<T> basis,
        std::function<std::optional<std::vector<F>>(const T&)> encode,
        std::function<T(const std::vector<F>&)> decode
    );

    // =====================================================
    // 基本向量运算
    // =====================================================

    T zero() const;
    T add(const T& x, const T& y) const;
    T subtract(const T& x, const T& y) const;
    T scalar_multiply(const F& a, const T& x) const;

    bool equal(const T& x, const T& y) const;
    bool is_zero(const T& x) const;
    bool contains(const T& x) const;

    std::size_t dim() const;
    const std::vector<T>& basis() const;

    std::optional<std::vector<F>>
    coordinates(const T& x) const;

    T vector_from_coordinates(const std::vector<F>& coordinates) const;

    // =====================================================
    // 线性无关、生成与换基
    // =====================================================

    bool is_linearly_independent(const std::vector<T>& vectors) const;
    bool is_spanning(const std::vector<T>& vectors) const;

    Self span(const std::vector<T>& vectors) const;

    Matrix change_of_basis(
        const std::vector<T>& from_basis,
        const std::vector<T>& to_basis
    ) const;

    // =====================================================
    // 子空间
    // =====================================================

    bool is_subspace(const Self& subspace) const;

    Self sum(const Self& other) const;
    Self intersection(const Self& other) const;

    // 内部直和：仍然是同一元素类型 T
    std::optional<Self> internal_direct_sum(const Self& other) const;

    // 外部直和：元素类型为 pair<T,U>
    template<class U>
    static LinearSpace<F, std::pair<T, U>> external_direct_sum(
        const Self& left,
        const LinearSpace<F, U>& right
    );

    QuotientSpace quotient(const Self& subspace) const;

    // =====================================================
    // Hom(T,U) 也是一个线性空间
    // 其元素类型是 Map<U>
    // =====================================================

    template<class U>
    LinearSpace<F, Map<U>>
    hom(const LinearSpace<F, U>& codomain) const;

    LinearSpace<F, Endomorphism>
    endomorphisms() const;

    // =====================================================
    // 线性映射
    // =====================================================

    template<class U>
    U apply(const Map<U>& map, const T& value) const;

    template<class U>
    Self kernel(
        const Map<U>& map,
        const LinearSpace<F, U>& codomain
    ) const;

    template<class U>
    LinearSpace<F, U> image(
        const Map<U>& map,
        const LinearSpace<F, U>& codomain
    ) const;

    template<class U>
    std::size_t rank(
        const Map<U>& map,
        const LinearSpace<F, U>& codomain
    ) const;

    template<class U>
    std::size_t nullity(
        const Map<U>& map,
        const LinearSpace<F, U>& codomain
    ) const;

    template<class U>
    Matrix matrix_of(
        const Map<U>& map,
        const LinearSpace<F, U>& codomain
    ) const;

    template<class U>
    bool is_injective(
        const Map<U>& map,
        const LinearSpace<F, U>& codomain
    ) const;

    template<class U>
    bool is_surjective(
        const Map<U>& map,
        const LinearSpace<F, U>& codomain
    ) const;

    template<class U>
    bool is_isomorphism(
        const Map<U>& map,
        const LinearSpace<F, U>& codomain
    ) const;

    std::optional<Endomorphism>
    inverse(const Endomorphism& map) const;

    // =====================================================
    // 对偶
    // =====================================================

    DualSpace dual() const;

    template<class U>
    DualMap<U> dual_map(
        const Map<U>& map,
        const LinearSpace<F, U>& codomain
    ) const;

    // 自然映射 V -> V**
    typename DualSpace::template Map<F>
    into_double_dual(const T& value) const;

    DualSpace annihilator(const Self& subspace) const;

    // =====================================================
    // 张量积
    // =====================================================

    template<class U>
    TensorSpace<U> tensor_product(
        const LinearSpace<F, U>& other
    ) const;

    // =====================================================
    // 仅针对自同态
    // =====================================================

    F trace(const Endomorphism& map) const;
    F determinant(const Endomorphism& map) const;

    std::vector<F>
    characteristic_polynomial(const Endomorphism& map) const;

    Endomorphism apply_polynomial(
        const std::vector<F>& coefficients,
        const Endomorphism& map
    ) const;

    Self eigenspace(
        const Endomorphism& map,
        const F& eigenvalue
    ) const;

    Self cyclic_subspace(
        const T& vector,
        const Endomorphism& map
    ) const;
};
```
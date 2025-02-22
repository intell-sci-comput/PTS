from nesymres.architectures.data_utils import (
    eq_constant_mul,
    eq_remap_pow,
    eq_remove_negative_integer_powers,
    eq_sympy_prefix_to_token_library,
)


def test_eq_remove_negative_integer_powers():
    assert eq_remove_negative_integer_powers(["pow", "x_2", "-2"]) == ["inv", "pow", "x_2", "2"]
    assert eq_remove_negative_integer_powers(["pow", "cos", "pow", "x_2", "2", "-2"]) == [
        "inv",
        "pow",
        "cos",
        "pow",
        "x_2",
        "2",
        "2",
    ]
    assert eq_remove_negative_integer_powers(["pow", "cos", "pow", "x_2", "-2", "-2"]) == [
        "inv",
        "pow",
        "cos",
        "inv",
        "pow",
        "x_2",
        "2",
        "2",
    ]
    assert eq_remove_negative_integer_powers(["mul", "x_1", "pow", "x_2", "-2"]) == [
        "mul",
        "x_1",
        "inv",
        "pow",
        "x_2",
        "2",
    ]
    assert eq_remove_negative_integer_powers(["pow", "cos", "x_2", "-2"]) == ["inv", "pow", "cos", "x_2", "2"]


def test_remap_pow():
    assert eq_remap_pow(["pow", "x_1", "2"]) == ["mul", "x_1", "x_1"]
    assert eq_remap_pow(["pow", "x_1", "3"]) == ["mul", "x_1", "mul", "x_1", "x_1"]
    assert eq_remap_pow(["pow", "cos", "x_1", "3"]) == ["mul", "cos", "x_1", "mul", "cos", "x_1", "cos", "x_1"]
    assert eq_remap_pow(["inv", "pow", "x_2", "2"]) == ["inv", "mul", "x_2", "x_2"]
    assert eq_remap_pow(["inv", "pow", "cos", "inv", "pow", "x_2", "2", "2"]) == [
        "inv",
        "mul",
        "cos",
        "inv",
        "mul",
        "x_2",
        "x_2",
        "cos",
        "inv",
        "mul",
        "x_2",
        "x_2",
    ]
    assert eq_remap_pow(["inv", "pow", "cos", "pow", "x_2", "2", "2"]) == [
        "inv",
        "mul",
        "cos",
        "mul",
        "x_2",
        "x_2",
        "cos",
        "mul",
        "x_2",
        "x_2",
    ]
    assert eq_remap_pow(["inv", "pow", "cos", "pow", "x_2", "3", "2"]) == [
        "inv",
        "mul",
        "cos",
        "mul",
        "x_2",
        "mul",
        "x_2",
        "x_2",
        "cos",
        "mul",
        "x_2",
        "mul",
        "x_2",
        "x_2",
    ]


def test_eq_constant_mul():
    assert eq_constant_mul(["mul", "2", "x_1"]) == ["add", "x_1", "x_1"]
    assert eq_constant_mul(["mul", "3", "x_1"]) == ["add", "x_1", "add", "x_1", "x_1"]
    assert eq_constant_mul(["mul", "4", "x_1"]) == ["add", "x_1", "add", "x_1", "add", "x_1", "x_1"]
    assert eq_constant_mul(["mul", "-2", "x_1"]) == ["add", "neg", "x_1", "neg", "x_1"]
    assert eq_constant_mul(["mul", "x_1", "2"]) == ["add", "x_1", "x_1"]
    assert eq_constant_mul(["mul", "x_1", "-2"]) == ["add", "neg", "x_1", "neg", "x_1"]
    assert eq_constant_mul(["mul", "2", "cos", "x_1"]) == ["add", "cos", "x_1", "cos", "x_1"]
    assert eq_constant_mul(["mul", "cos", "x_1", "3"]) == ["add", "cos", "x_1", "add", "cos", "x_1", "cos", "x_1"]
    assert eq_constant_mul(["mul", "sin", "mul", "2", "x_1", "3"]) == [
        "add",
        "sin",
        "add",
        "x_1",
        "x_1",
        "add",
        "sin",
        "add",
        "x_1",
        "x_1",
        "sin",
        "add",
        "x_1",
        "x_1",
    ]
    assert eq_constant_mul(["mul", "3", "sin", "mul", "2", "x_1"]) == [
        "add",
        "sin",
        "add",
        "x_1",
        "x_1",
        "add",
        "sin",
        "add",
        "x_1",
        "x_1",
        "sin",
        "add",
        "x_1",
        "x_1",
    ]
    assert eq_constant_mul(["mul", "3", "sin", "mul", "x_1", "2"]) == [
        "add",
        "sin",
        "add",
        "x_1",
        "x_1",
        "add",
        "sin",
        "add",
        "x_1",
        "x_1",
        "sin",
        "add",
        "x_1",
        "x_1",
    ]


def test_eq_sympy_prefix_to_token_library():
    assert eq_sympy_prefix_to_token_library(["pow", "x_2", "-2"]) == ["inv", "mul", "x2", "x2"]
    assert eq_sympy_prefix_to_token_library(["pow", "cos", "pow", "x_2", "2", "-2"]) == [
        "inv",
        "mul",
        "cos",
        "mul",
        "x2",
        "x2",
        "cos",
        "mul",
        "x2",
        "x2",
    ]
    assert eq_sympy_prefix_to_token_library(["pow", "cos", "pow", "x_2", "-2", "-2"]) == [
        "inv",
        "mul",
        "cos",
        "inv",
        "mul",
        "x2",
        "x2",
        "cos",
        "inv",
        "mul",
        "x2",
        "x2",
    ]
    assert eq_sympy_prefix_to_token_library(["mul", "x_1", "pow", "x_2", "-2"]) == [
        "mul",
        "x1",
        "inv",
        "mul",
        "x2",
        "x2",
    ]
    assert eq_sympy_prefix_to_token_library(["pow", "cos", "x_2", "-2"]) == ["inv", "mul", "cos", "x2", "cos", "x2"]
    assert eq_sympy_prefix_to_token_library(["pow", "x_1", "2"]) == ["mul", "x1", "x1"]
    assert eq_sympy_prefix_to_token_library(["pow", "x_1", "3"]) == ["mul", "x1", "mul", "x1", "x1"]
    assert eq_sympy_prefix_to_token_library(["pow", "cos", "x_1", "3"]) == [
        "mul",
        "cos",
        "x1",
        "mul",
        "cos",
        "x1",
        "cos",
        "x1",
    ]
    assert eq_sympy_prefix_to_token_library(["inv", "pow", "x_2", "2"]) == ["inv", "mul", "x2", "x2"]
    assert eq_sympy_prefix_to_token_library(["inv", "pow", "cos", "inv", "pow", "x_2", "2", "2"]) == [
        "inv",
        "mul",
        "cos",
        "inv",
        "mul",
        "x2",
        "x2",
        "cos",
        "inv",
        "mul",
        "x2",
        "x2",
    ]
    assert eq_sympy_prefix_to_token_library(["inv", "pow", "cos", "pow", "x_2", "2", "2"]) == [
        "inv",
        "mul",
        "cos",
        "mul",
        "x2",
        "x2",
        "cos",
        "mul",
        "x2",
        "x2",
    ]
    assert eq_sympy_prefix_to_token_library(["inv", "pow", "cos", "pow", "x_2", "3", "2"]) == [
        "inv",
        "mul",
        "cos",
        "mul",
        "x2",
        "mul",
        "x2",
        "x2",
        "cos",
        "mul",
        "x2",
        "mul",
        "x2",
        "x2",
    ]
    assert eq_sympy_prefix_to_token_library(["mul", "2", "x_1"]) == ["add", "x1", "x1"]
    assert eq_sympy_prefix_to_token_library(["mul", "3", "x_1"]) == ["add", "x1", "add", "x1", "x1"]
    assert eq_sympy_prefix_to_token_library(["mul", "4", "x_1"]) == ["add", "x1", "add", "x1", "add", "x1", "x1"]
    assert eq_sympy_prefix_to_token_library(["mul", "-2", "x_1"]) == ["add", "neg", "x1", "neg", "x1"]
    assert eq_sympy_prefix_to_token_library(["mul", "x_1", "2"]) == ["add", "x1", "x1"]
    assert eq_sympy_prefix_to_token_library(["mul", "x_1", "-2"]) == ["add", "neg", "x1", "neg", "x1"]
    assert eq_sympy_prefix_to_token_library(["mul", "2", "cos", "x_1"]) == ["add", "cos", "x1", "cos", "x1"]
    assert eq_sympy_prefix_to_token_library(["mul", "cos", "x_1", "3"]) == [
        "add",
        "cos",
        "x1",
        "add",
        "cos",
        "x1",
        "cos",
        "x1",
    ]
    assert eq_sympy_prefix_to_token_library(["mul", "sin", "mul", "2", "x_1", "3"]) == [
        "add",
        "sin",
        "add",
        "x1",
        "x1",
        "add",
        "sin",
        "add",
        "x1",
        "x1",
        "sin",
        "add",
        "x1",
        "x1",
    ]
    assert eq_sympy_prefix_to_token_library(["mul", "3", "sin", "mul", "2", "x_1"]) == [
        "add",
        "sin",
        "add",
        "x1",
        "x1",
        "add",
        "sin",
        "add",
        "x1",
        "x1",
        "sin",
        "add",
        "x1",
        "x1",
    ]
    assert eq_sympy_prefix_to_token_library(["mul", "3", "sin", "mul", "x_1", "2"]) == [
        "add",
        "sin",
        "add",
        "x1",
        "x1",
        "add",
        "sin",
        "add",
        "x1",
        "x1",
        "sin",
        "add",
        "x1",
        "x1",
    ]

    # Problematic expressions
    assert eq_sympy_prefix_to_token_library(
        ["add", "pow", "sin", "pow", "x_1", "3", "5", "mul", "-1", "pow", "x_2", "4"]
    ) == [
        "add",
        "mul",
        "sin",
        "mul",
        "x1",
        "mul",
        "x1",
        "x1",
        "mul",
        "sin",
        "mul",
        "x1",
        "mul",
        "x1",
        "x1",
        "mul",
        "sin",
        "mul",
        "x1",
        "mul",
        "x1",
        "x1",
        "mul",
        "sin",
        "mul",
        "x1",
        "mul",
        "x1",
        "x1",
        "sin",
        "mul",
        "x1",
        "mul",
        "x1",
        "x1",
        "neg",
        "mul",
        "x2",
        "mul",
        "x2",
        "mul",
        "x2",
        "x2",
    ]


if __name__ == "__main__":
    test_eq_remove_negative_integer_powers()
    test_remap_pow()
    test_eq_constant_mul()
    test_eq_sympy_prefix_to_token_library()
    print("Tests passed")

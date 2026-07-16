"""The shared strict-validation kernel is security surface: test every branch."""

import re

import pytest

from optima._strict import (
    duplicate_key_pairs,
    require_digest,
    require_driver_integer,
    require_exact_fields,
    require_identifier,
    require_int,
    truthy_flag,
)


class KernelTestError(RuntimeError):
    pass


VALID_DIGEST = "ab" * 32
ZERO_DIGEST = "0" * 64


class TestRequireDigest:
    def test_valid_digest_passes(self):
        assert require_digest(VALID_DIGEST, field="d", error=KernelTestError) == VALID_DIGEST

    @pytest.mark.parametrize(
        "value",
        [None, 7, b"ab" * 32, "AB" * 32, "ab" * 31, "ab" * 33, "xy" * 32, ""],
    )
    def test_malformed_rejected_with_caller_error(self, value):
        with pytest.raises(KernelTestError, match="lowercase 64-hex SHA-256"):
            require_digest(value, field="d", error=KernelTestError)

    def test_all_zero_rejected_by_default(self):
        with pytest.raises(KernelTestError, match="must not be the all-zero digest"):
            require_digest(ZERO_DIGEST, field="d", error=KernelTestError)

    def test_allow_zero_admits_the_zero_digest(self):
        assert (
            require_digest(ZERO_DIGEST, field="d", error=KernelTestError, allow_zero=True)
            == ZERO_DIGEST
        )

    def test_optional_admits_only_the_empty_string(self):
        assert require_digest("", field="d", error=KernelTestError, optional=True) == ""
        with pytest.raises(KernelTestError):
            require_digest(None, field="d", error=KernelTestError, optional=True)

    def test_optional_does_not_relax_zero_rejection(self):
        with pytest.raises(KernelTestError, match="all-zero"):
            require_digest(ZERO_DIGEST, field="d", error=KernelTestError, optional=True)

    def test_field_name_in_message(self):
        with pytest.raises(KernelTestError, match="my_field"):
            require_digest("nope", field="my_field", error=KernelTestError)


class TestRequireInt:
    def test_valid_int_passes(self):
        assert require_int(5, field="n", error=KernelTestError) == 5

    @pytest.mark.parametrize("value", [True, False, 5.0, "5", None])
    def test_non_exact_int_rejected(self, value):
        with pytest.raises(KernelTestError, match="n must be an integer"):
            require_int(value, field="n", error=KernelTestError)

    def test_minimum_enforced(self):
        assert require_int(0, field="n", error=KernelTestError, minimum=0) == 0
        with pytest.raises(KernelTestError, match=re.escape("integer >= 0")):
            require_int(-1, field="n", error=KernelTestError, minimum=0)

    def test_maximum_enforced(self):
        with pytest.raises(KernelTestError, match=re.escape("integer <= 4")):
            require_int(5, field="n", error=KernelTestError, maximum=4)

    def test_range_enforced_with_range_message(self):
        assert require_int(3, field="n", error=KernelTestError, minimum=1, maximum=4) == 3
        with pytest.raises(KernelTestError, match=re.escape("integer in [1, 4]")):
            require_int(0, field="n", error=KernelTestError, minimum=1, maximum=4)


class TestRequireIdentifier:
    PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,15}\Z")

    def test_valid_identifier_passes(self):
        assert (
            require_identifier("abc.1-x", field="i", error=KernelTestError, pattern=self.PATTERN)
            == "abc.1-x"
        )

    @pytest.mark.parametrize("value", [None, 7, "", "ABC", "-lead", "x" * 17])
    def test_off_grammar_rejected(self, value):
        with pytest.raises(KernelTestError, match="canonical identifier"):
            require_identifier(value, field="i", error=KernelTestError, pattern=self.PATTERN)


class TestRequireExactFields:
    FIELDS = frozenset({"a", "b"})

    def test_exact_mapping_passes(self):
        value = {"a": 1, "b": 2}
        assert (
            require_exact_fields(value, fields=self.FIELDS, label="row", error=KernelTestError)
            is value
        )

    def test_missing_and_extra_reported(self):
        with pytest.raises(KernelTestError, match=r"fields mismatch: missing=\('b',\), extra=\('c',\)"):
            require_exact_fields(
                {"a": 1, "c": 3}, fields=self.FIELDS, label="row", error=KernelTestError
            )

    def test_non_mapping_rejected(self):
        with pytest.raises(KernelTestError, match="must be an object"):
            require_exact_fields([1], fields=self.FIELDS, label="row", error=KernelTestError)

    def test_non_string_keys_rejected(self):
        with pytest.raises(KernelTestError, match="keys must be strings"):
            require_exact_fields(
                {"a": 1, 2: 2}, fields=self.FIELDS, label="row", error=KernelTestError
            )

    def test_exact_dict_rejects_mapping_stand_ins(self):
        import collections

        mapping = collections.OrderedDict(a=1, b=2)
        assert (
            require_exact_fields(mapping, fields=self.FIELDS, label="row", error=KernelTestError)
            is mapping
        )
        with pytest.raises(KernelTestError, match="must be a JSON object"):
            require_exact_fields(
                mapping, fields=self.FIELDS, label="row", error=KernelTestError, exact_dict=True
            )


class TestDuplicateKeyPairs:
    def test_unique_pairs_build_a_dict_in_order(self):
        assert duplicate_key_pairs(
            [("b", 2), ("a", 1)], label="doc", error=KernelTestError
        ) == {"b": 2, "a": 1}

    def test_duplicate_key_rejected(self):
        with pytest.raises(KernelTestError, match="doc contains duplicate key 'a'"):
            duplicate_key_pairs([("a", 1), ("a", 2)], label="doc", error=KernelTestError)


class _EnumLike:
    def __init__(self, value):
        self.value = value


class TestRequireDriverInteger:
    def test_plain_int_passes(self):
        assert require_driver_integer(7, field="status", error=KernelTestError) == 7

    def test_enum_value_path_passes(self):
        assert require_driver_integer(_EnumLike(3), field="status", error=KernelTestError) == 3

    @pytest.mark.parametrize(
        "value", [True, _EnumLike(True), _EnumLike("x"), "7", 7.5, None, object()]
    )
    def test_malformed_rejected(self, value):
        with pytest.raises(KernelTestError, match="malformed status"):
            require_driver_integer(value, field="status", error=KernelTestError)


class TestTruthyFlag:
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "On"])
    def test_truthy(self, value):
        assert truthy_flag(value) is True

    @pytest.mark.parametrize("value", [None, "", "0", "false", "off", "2", "enabled"])
    def test_falsy(self, value):
        assert truthy_flag(value) is False

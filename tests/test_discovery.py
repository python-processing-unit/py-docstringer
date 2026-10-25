from __future__ import annotations

from textwrap import dedent

from py_docstringer.discovery import discover_source


def test_discovers_only_undocumented_definitions() -> None:
    source = dedent(
        '''\
        def documented() -> None:
            """Already documented."""


        def plain(value: int) -> str:
            return str(value)


        async def asynchronous() -> None:
            ...


        class Documented:
            """Already documented."""

            def method(self) -> int:
                return 1

            @property
            def value(self) -> int:
                return 2

            class Inner:
                def nested_method(self) -> None:
                    ...


        class Undocumented:
            pass
        '''
    )

    definitions = discover_source(source)
    found = {(definition.qualname, definition.kind) for definition in definitions}

    assert found == {
        ("plain", "function"),
        ("asynchronous", "function"),
        ("Documented.method", "method"),
        ("Documented.value", "method"),
        ("Documented.Inner", "class"),
        ("Documented.Inner.nested_method", "method"),
        ("Undocumented", "class"),
    }


def test_can_include_documented_definitions() -> None:
    source = dedent(
        '''\
        def documented() -> None:
            """Already documented."""


        def undocumented() -> None:
            ...


        class Container:
            """Already documented."""

            def method(self) -> None:
                ...
        '''
    )

    definitions = discover_source(source, include_documented=True)
    found = {
        (definition.qualname, definition.kind, definition.has_docstring)
        for definition in definitions
    }

    assert found == {
        ("documented", "function", True),
        ("undocumented", "function", False),
        ("Container", "class", True),
        ("Container.method", "method", False),
    }


def test_discovers_nested_functions_and_tracks_method_context() -> None:
    source = dedent(
        """\
        def outer() -> None:
            def inner() -> None:
                ...

            return inner


        class Container:
            def method(self) -> None:
                def local() -> None:
                    ...

                return local

            class Nested:
                def nested_method(self) -> None:
                    ...
        """
    )

    definitions = discover_source(source)
    found = {(definition.qualname, definition.kind) for definition in definitions}

    assert found == {
        ("outer", "function"),
        ("outer.inner", "function"),
        ("Container", "class"),
        ("Container.method", "method"),
        ("Container.method.local", "function"),
        ("Container.Nested", "class"),
        ("Container.Nested.nested_method", "method"),
    }


def test_source_includes_decorators_and_full_body() -> None:
    source = dedent(
        '''\
        class Example:
            """Example."""

            @staticmethod
            def decorated() -> int:
                return 1
        '''
    )

    [definition] = discover_source(source)

    assert definition.qualname == "Example.decorated"
    assert definition.lineno == 5
    assert definition.source == (
        "    @staticmethod\n    def decorated() -> int:\n        return 1\n"
    )

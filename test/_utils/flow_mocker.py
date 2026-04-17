"""Utility for mocking flow of the test."""

from unittest.mock import PropertyMock


class FlowMocker:
    """Flow mocker.

    Mocker for the flow of the test. It allows creating mock objects and asserting them.

    Parameters
    ----------
    mocker : pytest_mock.plugin.MockerFixture
        Mocker object.
    params : list
        List of parameters.
    setup : bool, optional
        If True, setup mocker.
    """
    def __init__(self, mocker, params, setup=True):
        self.params = params
        self.mocker = mocker

        if setup:
            self.setup()

    def parse_mock(self, params):
        """Parse mock object from parameters.

        Parameters
        ----------
        params : dict
            Parameters.
        """
        for k in list(params.keys()):
            if "mock_" in k:
                params["mock"] = params[k]
                params["alias"] = k

    def setup(self):
        """Setup mocker."""
        mocker = self.mocker
        self.mocks = dict()

        for p in self.params:
            self.parse_mock(p)
            patch_obj = p.get("mock", None)
            mock_alias = p.get("alias", None)
            init_args = p.get("init_args", dict())

            if patch_obj is not None:
                if type(patch_obj) == tuple:
                    # Mocker is object
                    if "property_val" in init_args:
                        init_args["return_value"] = init_args["property_val"]
                        init_args["new_callable"] = PropertyMock
                        del init_args["property_val"]
                    # Create mock object for the instance
                    mock_obj = mocker.patch.object(*patch_obj, **init_args)
                elif type(patch_obj) == str:
                    # Create mock object for class method
                    mock_obj = mocker.patch(patch_obj, **init_args)
                else:
                    raise ValueError(f"Wrong patch object: {patch_obj}")

                self.mocks[patch_obj] = mock_obj
                if mock_alias is not None:
                    if mock_alias in self.mocks:
                        raise KeyError(
                            f"`{mock_alias}` already exists in mock dictionary"
                        )
                    self.mocks[mock_alias] = self.mocks[patch_obj]

    def assert_all(self, tested_obj, params=None):
        """Assert test.

        Assert that specified in assert_args either `params` or self.params. We can access mock object by its alias {"mock: "mock_alias_1", ...}.

        Parameters
        ----------
        tested_obj : any
            Tested object.
        params : list, optional
            List of parameters to assert. If None, self.params will be used.
        """
        params = params if params is not None else self.params

        for i, p in enumerate(self.params):
            mock_name = p.get("mock", None)
            assert_args = p.get("assert_args", None)

            if assert_args is not None:
                assert_func, *func_params = assert_args
                mock = self.mocks[mock_name] if mock_name is not None else None

                if hasattr(mock, f"assert_{assert_func}"):
                    assert_func = getattr(mock, f"assert_{assert_func}")
                    assert_func(*func_params)
                elif assert_func == "created_property":
                    if len(func_params) > 1:
                        raise ValueError(
                            "Expected one value for `created_property`"
                        )
                    assert hasattr(tested_obj, func_params[0])

    def get(self, mock_key):
        """Get mock object by its key.

        Parameters
        ----------
        mock_key : str
            Key of the mock object.

        Returns
        -------
        any
            Mock object.
        """
        return self.mocks[mock_key]

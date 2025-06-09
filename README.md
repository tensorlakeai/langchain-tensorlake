# LangChain Tensorlake

This repository contains 1 package with Tensorlake integrations with LangChain:

- [langchain-tensorlake](https://pypi.org/project/langchain-tensorlake/)

## Creating a TestPyPI version
This repository has a `TEST_PYPI_TOKEN` for publishing to [test.pypi.org](https://https://test.pypi.org/project/langchain-tensorlake/).

### Through the GitHub Workflow
To trigger a build and automatic publish to [test.pypi.org/project/langchain-tensorlake](https://test.pypi.org/project/langchain-tensorlake/), follow these steps:
1. Create a new git tag with `git tag v0.1.0`, updating the version number appropriately
2. Push the tag to GitHub with `git push origin v0.1.0`

### Manually
To manually build and publish to [test.pypi.org/project/langchain-tensorlake](https://test.pypi.org/project/langchain-tensorlake/), follow these steps:
1. Bump the version in [pyproject.toml](pyproject.toml).
2. Build with `python -m build`
3. Run `twine upload --repository testpypi dist/*`
4. Verify it was updated on [TestPyPI](https://test.pypi.org/project/langchain-tensorlake/)
5. Verify the installation with:  
    ```
    pip install --index-url https://test.pypi.org/simple \
                --extra-index-url https://pypi.org/simple \
                langchain-tensorlake
    ```

## Creating a PyPI verison
This repository has a `PYPI_API_TOKEN` for publishing to [pypi.org](https://https://pypi.org/project/langchain-tensorlake/).

### Through the GitHub Workflow
To trigger a build and automatic publish to [pypi.org/project/langchain-tensorlake](https://pypi.org/project/langchain-tensorlake/), follow these steps:
1. On GitHub, go to [Releases -> Create a new release](https://github.com/tensorlakeai/langchain-tensorlake/releases) 
2. Add a tag version, release title, and release description
3. Click Publish release

### Manually
To manually build and publish to [pypi.org/project/langchain-tensorlake](https://pypi.org/project/langchain-tensorlake/), follow these steps:
1. Bump the version in [pyproject.toml](pyproject.toml).
2. Build with `python -m build`
3. Run `twine upload dist/*`
4. Verify it was updated on [PyPI](https://pypi.org/project/langchain-tensorlake/)
5. Verify the installation with:  
    ```
    pip install langchain-tensorlake
    ```

### Clean Up After a Build

While testing, you may need to clean your environment. Make sure you remove all dist, build, and egg-info files:
```
rm -rf dist/ build/ src/langchain_tensorlake.egg-info
``` 
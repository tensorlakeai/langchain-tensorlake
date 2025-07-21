## Deploy Updated Package
1. Update the version in [`pyproject.toml`](https://github.com/tensorlakeai/langchain-tensorlake/blob/main/pyproject.toml)
2. Make sure everything is merged to `main`
3. Publish to Test PyPI:
  -  Create a tag: `git tag v.0.1.0`, updating the version number appropriately
  -  Push the tag to GitHub: `git push origin --tags`
  -  The [GitHub Action](https://github.com/tensorlakeai/langchain-tensorlake/actions/workflows/publish-testpypi.yaml) will kick off and publish to Test PyPI
  -  Confirm that the package on [test.pypi.org](https://test.pypi.org/project/langchain-tensorlake/) is the correct version and is working
5. Publish to PyPI:
  - [Releases -> Create a new release](https://github.com/tensorlakeai/langchain-tensorlake/releases)
  - Add a tag version, release title, and release description
  - Click Publish release
  - The [GitHub Action](https://github.com/tensorlakeai/langchain-tensorlake/actions/workflows/publish-pypi.yaml) will kick off an publish to PyPI
  - Confirm that the package on [PyPI](https://pypi.org/project/langchain-tensorlake/a) is the correct version and is working
 
## Manually publish to TestPyPI
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

## Manually publish to PyPI
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

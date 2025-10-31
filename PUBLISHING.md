# Publishing to PyPI

This guide covers how to publish the ClassifAI Python package to PyPI.

## Prerequisites

1. **PyPI Account**: Create accounts on [PyPI](https://pypi.org) and [TestPyPI](https://test.pypi.org)
2. **Install build tools**:
   ```bash
   pip install build twine
   ```

## Publishing Steps

### 1. Update Version

Edit `pyproject.toml` and increment the version:
```toml
version = "0.1.1"  # Increment this
```

### 2. Build the Package

```bash
cd classifai-python
python -m build
```

This creates:
- `dist/classifai-sdk-0.1.0.tar.gz` (source distribution)
- `dist/classifai-sdk-0.1.0-py3-none-any.whl` (wheel distribution)

### 3. Test on TestPyPI (Recommended)

Upload to TestPyPI first to catch any issues:

```bash
python -m twine upload --repository testpypi dist/*
```

Enter your TestPyPI credentials when prompted.

Test the installation:
```bash
pip install --index-url https://test.pypi.org/simple/ classifai-sdk
```

### 4. Publish to PyPI

Once tested, upload to the real PyPI:

```bash
python -m twine upload dist/*
```

Enter your PyPI credentials.

### 5. Verify Installation

```bash
pip install classifai-sdk
```

Test it:
```python
from classifai import ClassifAI
client = ClassifAI()
print(client.health_check())
```

## Using API Tokens (Recommended)

Instead of username/password, use API tokens for security:

1. Go to PyPI Account Settings → API tokens
2. Create a new token with "Entire account" or specific project scope
3. Create `~/.pypirc`:

```ini
[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmc...your-token-here...

[testpypi]
username = __token__
password = pypi-AgENdGVzdC5weXBpLm9yZw...your-token-here...
```

Then upload with:
```bash
python -m twine upload dist/*
```

## Automated Publishing with GitHub Actions

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.x'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine
      - name: Build package
        run: python -m build
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
```

Add `PYPI_API_TOKEN` to your GitHub repository secrets.

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR** version (1.0.0): Incompatible API changes
- **MINOR** version (0.1.0): Add functionality (backwards compatible)
- **PATCH** version (0.0.1): Bug fixes (backwards compatible)

## Checklist Before Publishing

- [ ] Update version in `pyproject.toml`
- [ ] Update CHANGELOG.md with changes
- [ ] Run tests (if you have them)
- [ ] Build package: `python -m build`
- [ ] Test on TestPyPI first
- [ ] Upload to PyPI: `python -m twine upload dist/*`
- [ ] Create GitHub release/tag
- [ ] Verify installation: `pip install classifai-sdk`
- [ ] Update documentation if needed

## Troubleshooting

**Error: "File already exists"**
- You're trying to upload a version that already exists
- Increment the version number in `pyproject.toml`

**Error: "Invalid distribution"**
- Clean the `dist/` directory: `rm -rf dist/`
- Rebuild: `python -m build`

**Import errors after installation**
- Ensure `classifai/__init__.py` exports all necessary classes
- Check `pyproject.toml` package configuration

## Resources

- [Python Packaging Guide](https://packaging.python.org/)
- [PyPI Help](https://pypi.org/help/)
- [Twine Documentation](https://twine.readthedocs.io/)

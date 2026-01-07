# Contributing to PET-CT Radiomics Pipeline

Thank you for your interest in contributing to this project!

## How to Contribute

### Reporting Issues

1. Check if the issue already exists in [GitHub Issues](https://github.com/haga-aki/pet-ct-radiomics-pipeline/issues)
2. If not, create a new issue with:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version, GPU)
   - Relevant log output

### Suggesting Enhancements

1. Open a GitHub Issue with the "enhancement" label
2. Describe the feature and its use case
3. Discuss before implementing major changes

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Commit with clear messages
7. Push to your fork
8. Open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/pet-ct-radiomics-pipeline.git
cd pet-ct-radiomics-pipeline

# Create development environment
conda env create -f environment.yml
conda activate pet_radiomics

# Install development dependencies
pip install pytest black flake8
```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions
- Keep functions focused and small

## Testing

```bash
# Run tests
pytest tests/

# Check code style
flake8 *.py
black --check *.py
```

## Documentation

- Update README.md for user-facing changes
- Update docs/ for technical changes
- Add inline comments for complex logic

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to open an issue for any questions about contributing.

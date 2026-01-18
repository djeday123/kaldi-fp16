# Contributing to Kaldi-FP16

Thank you for your interest in contributing to Kaldi-FP16! This document provides guidelines for contributing to the project.

## 🌟 Ways to Contribute

- **Bug Reports**: Report bugs through GitHub issues
- **Feature Requests**: Suggest new features or improvements
- **Code Contributions**: Submit pull requests with bug fixes or new features
- **Documentation**: Improve documentation, examples, or tutorials
- **Testing**: Test on different GPU architectures and report results

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/kaldi-fp16.git
   cd kaldi-fp16
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 💻 Development Setup

### Prerequisites

- NVIDIA GPU with Tensor Core support (Compute Capability 7.0+)
- CUDA Toolkit 11.0 or later
- CMake 3.18 or later
- C++14 compatible compiler

### Building

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

### Running Tests

```bash
cd build
ctest --output-on-failure
```

## 📝 Code Style

- Follow C++14 standard
- Use 2 spaces for indentation
- Maximum line length: 100 characters
- Use descriptive variable and function names
- Add comments for complex algorithms

### Naming Conventions

- Classes: `PascalCase` (e.g., `MatrixFP16`)
- Functions: `PascalCase` (e.g., `AddMatMat`)
- Variables: `snake_case` (e.g., `num_rows_`)
- Constants: `kPascalCase` (e.g., `kNoTrans`)
- Private members: trailing underscore (e.g., `data_`)

### Example

```cpp
class MatrixFP16 {
 public:
  void AddMatMat(float alpha, const MatrixFP16& A);
  
 private:
  size_t num_rows_;
  half* data_;
};
```

## 🧪 Testing Guidelines

- Add unit tests for new features
- Ensure all tests pass before submitting PR
- Test on multiple GPU architectures if possible
- Include performance benchmarks for optimization changes

## 📄 Documentation

- Update README.md for user-facing changes
- Add inline comments for complex code
- Update API documentation for new public methods
- Include usage examples for new features

## 🔧 Pull Request Process

1. **Update your branch** with the latest main:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Commit your changes** with clear messages:
   ```bash
   git commit -m "Add feature: brief description"
   ```

3. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create a Pull Request** on GitHub

5. **Address review feedback** promptly

### PR Requirements

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages are clear and descriptive
- [ ] No merge conflicts with main branch

## 🐛 Bug Reports

When filing a bug report, please include:

- **Description**: Clear description of the bug
- **Steps to reproduce**: Minimal steps to reproduce the issue
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Environment**:
  - GPU model and compute capability
  - CUDA version
  - Operating system
  - Compiler version
- **Logs/Screenshots**: Any relevant error messages or logs

## 💡 Feature Requests

When suggesting a feature:

- Clearly describe the feature and its benefits
- Explain the use case
- Provide examples if possible
- Consider implementation complexity

## 🌍 Multilingual Contributions

We welcome documentation contributions in multiple languages:
- English (primary)
- Turkish
- Azerbaijani
- Russian
- Other languages welcome!

## 📧 Contact

- GitHub Issues: For bugs and feature requests
- Discussions: For questions and general discussions

## 📜 License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

## 🙏 Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect differing viewpoints and experiences

Thank you for contributing to Kaldi-FP16! 🚀

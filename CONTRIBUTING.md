# Contributing to Hyper-Mnemosyne

Thank you for your interest in contributing! This document provides guidelines for development.

## Getting Started

1. **Fork & Clone**

   ```bash
   git clone https://github.com/yourusername/Hyper-Mnemosyne.git
   cd Hyper-Mnemosyne
   ```

2. **Install Development Environment**

   ```bash
   pip install -e ".[dev]"
   ```

3. **Run Tests**

   ```bash
   python test_training.py
   ```

## Development Workflow

### Making Changes

1. Create a feature branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes

3. Test locally:

   ```bash
   python test_training.py
   ```

4. Commit with descriptive messages:

   ```bash
   git commit -m "Add feature: description"
   ```

5. Push and create a PR:

   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings to functions
- Keep functions focused and modular

## Testing

Before submitting a PR:

1. Run the test suite:

   ```bash
   python test_training.py
   ```

2. Verify training works:

   ```bash
   python training/train.py --data_dir data/test/ --max_steps 100 --training_stage backbone
   ```

3. Check for common issues:
   - No hardcoded paths
   - Imports work without PYTHONPATH hacks
   - Works on fresh clone

## Areas for Contribution

### High Priority

- [ ] Add comprehensive unit tests
- [ ] Implement validation metrics (perplexity, accuracy)
- [ ] Add distributed training support (DDP)
- [ ] Optimize memory usage for larger models

### Features

- [ ] Add inference server (FastAPI/vLLM)
- [ ] Implement quantization (int8/int4)
- [ ] Add LoRA/QLoRA fine-tuning
- [ ] Create benchmarking suite

### Documentation

- [ ] Add architecture diagrams
- [ ] Write training guide with hyperparameter tips
- [ ] Create example notebooks
- [ ] Document model architecture decisions

## Questions?

Open an issue or start a discussion!

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

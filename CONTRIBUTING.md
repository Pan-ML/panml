# Contributing to PanML
You're very welcomed to contribute, and we will be excited to have you join us for the development journey! We also encourage you to check out [design intent of PanML](https://github.com/Pan-ML/panml/wiki/0.-Design-intent-of-PanML) to get an idea of roughly the direction we wanted to take with this project.

Our discord channel: https://discord.gg/QpquRMDq

### How to submit a contribution
To make a contribution, please adhere to the following steps:
1. Fork and clone this repository
2. Do the changes on your fork
3. If you have modified the code (refactor, feature or bug-fix) please add the necessary tests for it in the test folder
4. Ensure that all tests pass (see below)
5. Submit a pull request

For more details about pull requests, please read [GitHub's guides](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request).

### Testing
Make run tests and make sure that all tests pass before submitting a pull request.
```bash
python3 -m unittest
```

### Release Process
We intend to release on a manual basis depending on the features and issues that the release will be addressing. Please reach out to us on discord if you have any ideas or suggestions. Typically once a release is ready to be deployed, a developer with admin rights to the repository will create a new release on GitHub, and then publish the new version to PyPI.

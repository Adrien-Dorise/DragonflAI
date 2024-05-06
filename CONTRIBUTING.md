
# Contributing to DragonflAI

First off, thanks for taking the time to contribute! ❤️

All types of contributions are encouraged and valued. See the [Table of Contents](#table-of-contents) for different ways to help and details about how this project handles them. Please make sure to read the relevant section before making your contribution. It will make it a lot easier for us maintainers and smooth out the experience for all involved. The community looks forward to your contributions. 🎉

> And if you like the project, but just don't have time to contribute, that's fine. There are other easy ways to support the project and show your appreciation, which we would also be very happy about:
> - Star the project
> - Tweet about it
> - Refer this project in your project's readme
> - Mention the project at local meetups and tell your friends/colleagues


## Table of Contents
- [DragonflAI description](#dragonflai-description)
- [Code of Conduct](#code-of-conduct)
- [I Have a Question](#i-have-a-question)
- [I Want To Contribute](#i-want-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Your First Code Contribution](#your-first-code-contribution)

## DragonflAI description  

DragonflAI is an open source API allowing user to develop artificial intelligence's based project.  
this API provide most of low level function and method used during the process of AI model developpement.  

Imagine an experiment, dragonflAI provide results  
1. Define data's experiment  
2. Select an available model or create your own  
3. Train your model  
4. Visualize results  
5. Compare results  
6. Save experiment

## Code of Conduct

This project and everyone participating in it is governed by the
[DragonflAI Code of Conduct](/CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code. Please report unacceptable behavior
to <>.


## I Have a Question

> If you want to ask a question, we assume that you have read the available [Documentation](https://gitlab.com/lr-technologies2/dragonflai/-/blob/main/docs/submodule_howto.md?ref_type=heads).

Before you ask a question, it is best to search for existing [Issues](https://gitlab.com/lr-technologies2/dragonflai//issues) that might help you. In case you have found a suitable issue and still need clarification, you can write your question in this issue. It is also advisable to search the internet for answers first.

If you then still feel the need to ask a question and need clarification, we recommend the following:

- Open an [Issue](https://gitlab.com/lr-technologies2/dragonflai//issues/new).
- Provide as much context as you can about what you're running into.
- Provide project and platform versions depending on what seems relevant.
- Select label <span style="color:purple">**Question**</span>.

We will then take care of the issue as soon as possible.


## I Want To Contribute

> ### Legal Notice 
> When contributing to this project, you must agree that you have authored 100% of the content, that you have the necessary rights to the content and that the content you contribute may be provided under the project license.

### Reporting Bugs


#### Before Submitting a Bug Report

A good bug report shouldn't leave others needing to chase you up for more information. Therefore, we ask you to investigate carefully, collect information and describe the issue in detail in your report. Please complete the following steps in advance to help us fix any potential bug as fast as possible.

- Make sure that you are using the latest version.
- Determine if your bug is really a bug and not an error on your side e.g. using incompatible environment components/versions (Make sure that you have read the [documentation](https://gitlab.com/lr-technologies2/dragonflai/-/blob/main/docs/submodule_howto.md?ref_type=heads). If you are looking for support, you might want to check [this section](#i-have-a-question)).
- To see if other users have experienced (and potentially already solved) the same issue you are having, check if there is not already a bug report existing for your bug or error in the [bug tracker](https://gitlab.com/lr-technologies2/dragonflai/-/issues/?sort=created_date&label_name%5B%5D=Bug).
- Also make sure to search the internet (including Stack Overflow) to see if users outside of the GitLab community have discussed the issue.
- Collect information about the bug:
  - Stack trace (Traceback)
  - OS, Platform and Version (Windows, Linux, macOS, x86, ARM)
  - Version of the interpreter, compiler, SDK, runtime environment, package manager, depending on what seems relevant.
  - Possibly your input and the output
  - Can you reliably reproduce the issue? And can you also reproduce it with older versions?


#### How Do I Submit a Good Bug Report?

> You must never report security related issues, vulnerabilities or bugs including sensitive information to the issue tracker, or elsewhere in public. Instead sensitive bugs must be sent by email to <>.

We use GitLab issues to track bugs and errors. If you run into an issue with the project:

- Open an [Issue](https://gitlab.com/lr-technologies2/dragonflai/-/issues/new). (Since we can't be sure at this point whether it is a bug or not, we ask you not to talk about a bug yet and not to label the issue.)
- Select *bug_report* template into Description field.
- Complete all fields in the dedicated area.
- Provide the information you collected in the previous section.

Once it's filed:

- A team member will try to reproduce the issue with your provided steps. If there are no reproduction steps or no obvious way to reproduce the issue, the team will ask you for those steps.


### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for DragonflAI, **including completely new features and minor improvements to existing functionality**. Following these guidelines will help maintainers and the community to understand your suggestion and find related suggestions.


#### Before Submitting an Enhancement

- Make sure that you are using the latest version.
- Read the [documentation](https://gitlab.com/lr-technologies2/dragonflai/-/blob/main/docs/submodule_howto.md?ref_type=heads) carefully and find out if the functionality is already covered, maybe by an individual configuration.
- Perform a [search](https://gitlab.com/lr-technologies2/dragonflai/-/issues/?sort=created_date&label_name%5B%5D=Feature) to see if the enhancement has already been suggested. If it has, add a comment to the existing issue instead of opening a new one.
- Find out whether your idea fits with the scope and aims of the project. It's up to you to make a strong case to convince the project's developers of the merits of this feature. Keep in mind that we want features that will be useful to the majority of our users and not just a small subset. If you're just targeting a minority of users, consider writing an add-on/plugin library.


#### How Do I Submit a Good Enhancement Suggestion?

Enhancement suggestions are tracked as [GitLab issues](https://gitlab.com/lr-technologies2/dragonflai/-/issues).

- Use a **clear and descriptive title** for the issue to identify the suggestion.
- Select *feature_request* template into Description field.
- Complete all fields in the dedicated area.
- **Explain why this enhancement would be useful** to most DragonflAI users. You may also want to point out the other projects that solved it better and which could serve as inspiration.

### Your First Code Contributio

You found an opened issue and you want to develop code in order to close this issue ?  

1. Make sure the issue is approved and no merge request opened  
2. Clone dragonflAI repository on your personnal computer  
3. Use at least python 3.10 version   
4. Create a virtual environment : `python3 -m venv venv_Name` 
5. Activate your venv : `source venv_Name/bin/activate`  
6. Install dependencies : `pip3 install -r docs/requirement.txt`  
7. Open a merge request on gitlab issue's page and set a branch name  
8. Switch to your branch : `git pull` then `git checkout branch_Name`   
9. Develop your code  
10. Commit with usefull message and push : `git add file_Name.py` and `git commit -m "my Message"` then `git push -u origin branch_Name`  
11. Comment your merge request a request review when implementation is done  
12. Enjoy developping and using dragonflAI  

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a>
    <img src="images/icon.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Audio Classification</h3>

</p>

<!-- INTRODUCTION -->
## Introduction
This project is for Audio Classification using Transformer Architecture. The following are the expriments that are conducted -
1. Audio classification using Mel Spectograms(mels).
2. Application of ViT to the mels.
3. Application of Masked Auto Encoders.
4. Applicaiton of Swin Transformers.

<!-- GETTING STARTED -->
## Getting Started

Follow all the steps carefully

### Prerequisites

This project requires python 3.6 or greater and also use virutalenv
* Installing virtualenv
  ```sh
  sudo pip3 install virtualenv
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Saivivek-Peddi/audio_transformer.git
   ```
2. CD into the project directory
   ```sh
   cd audio_transformer
   ```
4. Create a virtualenv and start it
   ```sh
   virtualenv venv && source venv/bin/activate
   ```
5. Install all the requirements
   ```sh
   pip install -r requirements.txt
   ```


<!-- USAGE EXAMPLES -->
## Usage

<!-- 1. Open `api_config.json` and edit url and token ([Click here](https://www.youtube.com/watch?v=cZ5cn8stjM0) to know how to generate a canvas api token)
2. To collect the submissions and generate temporary grades for specific quizzes. Run -
   ```sh
   python get_subs_and_grades_main.py
   ```
3. One completed, the above script will generate a json with quiz name inside `manual_inspections` folder.
4. This folder will contain the information of all the flagged students along with the detailed reasoning for getting flagged.
5. You can edit the values of `score`,`flag` keys.
6. Please note that a comment will be posted if and only if `flag` is set to `true` and also by default all non-flagged students are set to `false`.
7. You can also add additional questions in the `quesitons` dictionary of `evaluation` key.
8. Once manual inspection is complete, to post the grades to canvas - Run -
   ```sh
   python post_grades_main.py
   ```
9. This script will post all the grades on to canvas and also generate files inside `final_grades` and `final_tags` fodler to list all final grades assigned and final tagged students for further processing. -->

<!-- DEMO -->
<!-- ## Demo -->
<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<!-- CONTACT -->
## Contact

Sai Vivek Peddi - svpeddi@ucdavis.edu

<!-- Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name) -->


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/Saivivek-Peddi/Zillow-Scraper/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/saivivek-peddi
[product-screenshot]: images/icon.png

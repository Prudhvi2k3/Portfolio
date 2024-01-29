import React from 'react';
import project1Image from './assets/project-1.png';
import project2Image from './assets/project-2.png';
import project3Image from './assets/project-3.png';

const Projects = () => {
  return (
    <section id="projects">
      <h1 className="title">Projects</h1>
      <div className="experience-details-container">
        <div className="about-containers">
          {/* Project 1 */}
          <ProjectDetails
            imgSrc={project1Image}
            alt="Project 1"
            title="Iris Flower Classification"
            githubLink="https://github.com/Prudhvi2k3/IrisFlowerCLassification"
          />

          {/* Project 2 */}
          <ProjectDetails
            imgSrc={project2Image}
            alt="Project 2"
            title="Movie Rating"
            githubLink="https://github.com/Prudhvi2k3"
          />

          {/* Project 3 */}
          <ProjectDetails
            imgSrc={project3Image}
            alt="Project 3"
            title="Data Visualization"
            githubLink="https://github.com/Prudhvi2k3/DataVisualizer"
          />
        </div>
      </div>
    </section>
  );
};

const ProjectDetails = ({ imgSrc, alt, title, githubLink }) => (
  <div className="details-container color-container">
    <div className="article-container">
      <img src={imgSrc} alt={alt} className="project-img" />
    </div>
    <h2 className="experience-sub-title project-title">{title}</h2>
    <div className="btn-container">
      <button
        className="btn btn-color-2 project-btn"
        onClick={() => window.location.href = githubLink}
      >
        Github
      </button>
    </div>
  </div>
);

export default Projects;

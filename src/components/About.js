import React from 'react';
import Picture from './assets/about-pic.png'

const About = () => {
  return (
    <section id="about">
      <h1 className="title">About Me</h1>
      <div className="section-container">
        <div className="section__pic-container">
          <img
            src={Picture}
            alt="Profile pic"
            className="about-pic"
          />
        </div>
        <div className="about-details-container">
          <div className="about-containers">
            <div className="details-container">
              <h3>Experience</h3>
              <p>Fresher <br />BackEnd Developer</p>
            </div>
            <div className="details-container">
              <h3>Education</h3>
              <p>B.Tech CSD<br />Data Science</p>
            </div>
          </div>
          <div className="text-container">
            <p>
              Currently Pursuing a Bachelor's degree in Computer Science with a specialization in Data Science,
              I am deeply passionate about the dynamic fields of Data Science, Full Stack Development, and Machine Learning.
              My academic journey has equipped me with a solid foundation in computer science principles, 
              and I have eagerly delved into the world of data analytics, programming, and AI during my coursework. 
              I am driven by the potential to harness data-driven insights to solve real-world problems, 
              and I have honed my skills through hands-on projects and extracurricular activities. 
              With a strong commitment to continuous learning and innovation.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
};

export default About;

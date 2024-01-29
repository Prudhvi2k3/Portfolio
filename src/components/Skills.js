import React from 'react';
import darkcheckmark from './assets/checkmark_dark.png'

const Skills = () => {
  return (
    <section id="skills">
      <h1 className="title">Skills</h1>
      <div className="experience-details-container">
        <div className="about-containers">
          {/* Technical Skills */}
          <div className="details-container">
            <h2 className="experience-sub-title">Technical Skills</h2>
            <div className="article-container">
              <SkillArticle   iconDarkSrc={darkcheckmark} title="HTML"  />
              <SkillArticle   iconDarkSrc={darkcheckmark} title="CSS"  />
              <SkillArticle   iconDarkSrc={darkcheckmark} title="Python"  />
              <SkillArticle   iconDarkSrc={darkcheckmark} title="JavaScript"  />
              <SkillArticle   iconDarkSrc={darkcheckmark} title="C Language"  />
            </div>
          </div>

          {/* Soft Skills */}
          <div className="details-container">
            <h2 className="experience-sub-title">Soft Skills</h2>
            <div className="article-container">
              <SoftSkillArticle   iconDarkSrc={darkcheckmark} title="Adaptability" />
              <SoftSkillArticle   iconDarkSrc={darkcheckmark} title="Communication Skills" />
              <SoftSkillArticle   iconDarkSrc={darkcheckmark} title="Presentation Skills" />
              <SoftSkillArticle  iconDarkSrc={darkcheckmark} title="Team Work" />
              <SoftSkillArticle  iconDarkSrc={darkcheckmark} title="Collaboration & Team Work" />
            </div>
          </div>
        </div>  
      </div>
    </section>
  );
};

const SkillArticle = ({ iconSrc, iconLightSrc, iconDarkSrc, title, level }) => (
  <article>
    <img
      src={iconSrc}
      srcSet={`${iconLightSrc} 1x, ${iconDarkSrc} 2x`}
      alt="Experience icon"
      className="icon"
    />
    <div>
      <h3>{title}</h3>
    </div>
  </article>
);

const SoftSkillArticle = ({ iconSrc, iconLightSrc, iconDarkSrc, title }) => (
  <article>
    <img
      src={iconSrc}
      srcSet={`${iconLightSrc} 1x, ${iconDarkSrc} 2x`}
      alt="Experience icon"
      className="icon"
    />
    <div>
      <h3>{title}</h3>
    </div>
  </article>
);

export default Skills;

import React from 'react';

const Contact = () => {
  return (
    <section id="contact">
      <h1 className="title">Contact Me</h1>
      <div className="contact-info-upper-container">
        <ContactInfo
          alt="Email icon"
          link="mailto:prudhviankamreddi1@gmail.com"
          text="Gmail"
        />

        <ContactInfo
          alt="LinkedIn icon"
          link="https://www.linkedin.com/in/prudhvi-ankamreddi/"
          text="LinkedIn"
        />
      </div>
    </section>
  );
};

const ContactInfo = ({ iconSrc, alt, link, text }) => {
  return (
    <div className="contact-info-container">
      <p><a href={link}>{text}</a></p>
    </div>
  );
};

export default Contact;

import React from 'react';
import ProfilepicDark from './assets/profile-pic.png';
import PrudhviCV from './assets/Prudhvi.pdf';

const Profile = () => {
  const openCV = () => {
    window.open(PrudhviCV);
  };
  return (
    <section id="profile">
      <div className="section__pic-container">
        <img src={ProfilepicDark} alt="profile pic" />
      </div>
      <div className="section__text">
        <p className="section__text__p1">Hello, I'm</p>
        <h1 className="title">Prudhvi Ankamreddi</h1>
        <p className="section__text__p2">BackEnd Developer</p>
        <div className="btn-container">
          <button
            className="btn btn-color-2"
            onClick={openCV}
          >
            Download CV
          </button>
          {/* Uncomment the line below if you want to include the Contact Info button */}
          {/* <button className="btn btn-color-1" onClick={() => (window.location.href = 'https://wa.me/qr/5AWO3EAXRJJ3C1')}>
            Contact Info
          </button> */}
        </div>
      </div>
    </section>
  );
};

export default Profile;

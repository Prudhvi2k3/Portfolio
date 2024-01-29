// Header.js
import React, { useEffect } from 'react';
import '../App.css';

const Header = () => {
  useEffect(() => {
    // Implement your dark mode logic here
    const themeIcons = document.querySelectorAll(".icon");
    setDarkMode(themeIcons);
    document.body.setAttribute("theme", "dark");
  }, []);

  const setDarkMode = icons => {
    icons.forEach(icon => {
      icon.src = icon.getAttribute("src-dark");
    });
  };

  const toggleMenu = () => {
    const menu = document.querySelector(".menu-links");
    const icon = document.querySelector(".hamburger-icon");
    menu.classList.toggle("open");
    icon.classList.toggle("open");
  };

  return (
    <header>
      <nav id="desktop-nav">
        <div className="logo">Prudhvi</div>
        <div>
          <ul className="nav-links">
            <li><a href="#about">About</a></li>
            <li><a href="#skills">Skills</a></li>
            <li><a href="#projects">Projects</a></li>
            <li><a href="#contact">Contact</a></li>
          </ul>
        </div>
      </nav>
      <HamburgerMenu toggleMenu={toggleMenu} />
    </header>
  );
};

const HamburgerMenu = ({ toggleMenu }) => {
  return (
    <nav id="hamburger-nav">
      <div className="logo">Prudhvi Ankamreddi</div>
      <div className="hamburger-menu">
        <div className="hamburger-icon" onClick={toggleMenu}>
          <span></span>
          <span></span>
          <span></span>
        </div>
        <div className="menu-links">
          <li><a href="#about" onClick={toggleMenu}>About</a></li>
          <li><a href="#skills" onClick={toggleMenu}>Skills</a></li>
          <li><a href="#projects" onClick={toggleMenu}>Projects</a></li>
          <li><a href="#contact" onClick={toggleMenu}>Contact</a></li>
        </div>
      </div>
    </nav>
  );
};

export default Header;

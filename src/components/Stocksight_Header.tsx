import React from "react";

interface HeaderProps {
  title: string;
}

const Header: React.FC<HeaderProps> = ({ title }) => {
  return (
    <div className="position-absolute top-0 start-50 translate-middle-x mt-4">
      <h1 className="fw-bold">{title}</h1>
    </div>
  );
};

export default Header;

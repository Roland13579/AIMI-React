import React, { useState } from "react";
import "bootstrap-icons/font/bootstrap-icons.css"; // Bootstrap Icons

interface VerificationPopupProps {
  email: string;
  onVerify: (code: string) => void;
  onClose: () => void;
}

const VerificationPopup: React.FC<VerificationPopupProps> = ({
  email,
  onVerify,
  onClose,
}) => {
  const [code, setCode] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    onVerify(code);
  };

  return (
    <div
      className="modal"
      style={{ display: "block", backgroundColor: "rgba(0, 0, 0, 0.7)" }}
    >
      <div className="modal-dialog modal-dialog-centered">
        <div className="modal-content border-0 shadow-lg" style={{ borderRadius: "10px" }}>
          <div className="modal-header border-0 bg-primary text-white py-2">
            <h5 className="modal-title" style={{ fontSize: "1.1rem" }}>
              <i className="bi bi-envelope-check me-2"></i>
              Verify Your Email
            </h5>
            <button
              type="button"
              className="btn-close btn-close-white"
              onClick={onClose}
            ></button>
          </div>
          <div className="modal-body p-4">
            <div className="text-center mb-4">
              <div className="bg-light rounded-circle mx-auto mb-3 d-flex align-items-center justify-content-center" 
                   style={{ width: "70px", height: "70px" }}>
                <i className="bi bi-shield-check text-primary" style={{ fontSize: "2rem" }}></i>
              </div>
              <p className="mb-1" style={{ fontSize: "0.9rem" }}>
                A 6-digit verification code has been sent to:
              </p>
              <p className="fw-bold text-primary mb-0" style={{ fontSize: "0.95rem" }}>{email}</p>
            </div>
            
            <form onSubmit={handleSubmit}>
              <div className="mb-4">
                <div className="input-group">
                  <span className="input-group-text bg-light border-end-0">
                    <i className="bi bi-key text-primary"></i>
                  </span>
                  <input
                    type="text"
                    className="form-control border-start-0 ps-0 text-center fw-bold"
                    placeholder="Enter 6-digit code"
                    value={code}
                    onChange={(e) => setCode(e.target.value)}
                    maxLength={6}
                    required
                    style={{ letterSpacing: "0.3em", fontSize: "1.1rem" }}
                  />
                </div>
                <div className="form-text text-center mt-2" style={{ fontSize: "0.75rem" }}>
                  <i className="bi bi-info-circle me-1"></i>
                  Please check your inbox and spam folder
                </div>
              </div>
              
              <div className="d-grid gap-2">
                <button type="submit" className="btn btn-primary py-2 fw-bold" style={{ fontSize: "1rem" }}>
                  <i className="bi bi-check-circle me-2"></i>
                  Verify Email
                </button>
                <button type="button" className="btn btn-outline-secondary" style={{ fontSize: "1rem" }} onClick={onClose}>
                  Verify Later
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VerificationPopup;

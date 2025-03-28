import React, { useState } from "react";

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
      style={{ display: "block", backgroundColor: "rgba(0, 0, 0, 0.5)" }}
    >
      <div className="modal-dialog">
        <div className="modal-content">
          <div className="modal-header">
            <h5 className="modal-title">Verify Your Email</h5>
            <button
              type="button"
              className="btn-close"
              onClick={onClose}
            ></button>
          </div>
          <div className="modal-body">
            <p>
              A 6-digit verification code has been sent to{" "}
              <strong>{email}</strong>.
            </p>
            <form onSubmit={handleSubmit}>
              <div className="mb-3">
                <label className="form-label">Verification Code</label>
                <input
                  type="text"
                  className="form-control"
                  value={code}
                  onChange={(e) => setCode(e.target.value)}
                  required
                />
              </div>
              <button type="submit" className="btn btn-primary">
                Verify
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VerificationPopup;

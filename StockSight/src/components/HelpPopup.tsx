import React from "react";
import { Modal, Button, Tab, Nav, Card } from "react-bootstrap";

interface HelpPopupProps {
  show: boolean;
  onHide: () => void;
  tabName: string;
  content: {
    features: React.ReactNode;
    calculations: React.ReactNode;
  };
}

const HelpPopup: React.FC<HelpPopupProps> = ({ show, onHide, tabName, content }) => {
  return (
    <Modal
      show={show}
      onHide={onHide}
      size="lg"
      centered
      className="help-popup"
    >
      <Modal.Header closeButton>
        <Modal.Title>
          <i className="bi bi-lightbulb-fill text-warning me-2"></i>
          {tabName} Help
        </Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <Tab.Container defaultActiveKey="features">
          <Nav variant="tabs" className="mb-3">
            <Nav.Item>
              <Nav.Link eventKey="features">Features</Nav.Link>
            </Nav.Item>
            <Nav.Item>
              <Nav.Link eventKey="calculations">Calculations & Definitions</Nav.Link>
            </Nav.Item>
          </Nav>
          <Tab.Content>
            <Tab.Pane eventKey="features">
              <Card body className="border-0">
                {content.features}
              </Card>
            </Tab.Pane>
            <Tab.Pane eventKey="calculations">
              <Card body className="border-0">
                {content.calculations}
              </Card>
            </Tab.Pane>
          </Tab.Content>
        </Tab.Container>
        <div className="text-center mt-3">
          <small className="text-muted">
            <i className="bi bi-robot me-1"></i>
            Powered by StockSight AI and SingStat API
          </small>
        </div>
      </Modal.Body>
      <Modal.Footer>
        <Button variant="primary" onClick={onHide}>
          Got it!
        </Button>
      </Modal.Footer>
    </Modal>
  );
};

export default HelpPopup;

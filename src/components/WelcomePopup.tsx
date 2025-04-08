import React from "react";
import { Modal, Button, Card, Row, Col } from "react-bootstrap";

interface WelcomePopupProps {
  show: boolean;
  onHide: () => void;
}

const WelcomePopup: React.FC<WelcomePopupProps> = ({ show, onHide }) => {
  return (
    <Modal
      show={show}
      onHide={onHide}
      size="lg"
      centered
      backdrop="static"
      className="welcome-popup"
    >
      <Modal.Header closeButton className="border-0 pb-0">
        <Modal.Title className="w-100 text-center">
          <h2 className="fw-bold text-primary">Welcome to StockSight</h2>
        </Modal.Title>
      </Modal.Header>
      <Modal.Body className="pt-0">
        <div className="text-center mb-3">
          <p className="lead" style={{ fontSize: "1.1rem" }}>
            Your intelligent inventory management and business forecasting solution
          </p>
          <p className="text-muted">
            <small>Powered by StockSight AI and SingStat API</small>
          </p>
        </div>

        <h5 className="text-center mb-3" style={{ fontSize: "1.1rem" }}>Discover what StockSight can do for you</h5>

        <Row className="g-3">
          <Col md={6}>
            <Card className="h-100 shadow-sm border-primary border-top border-4">
              <Card.Body>
                <div className="d-flex align-items-center mb-2">
                  <div className="rounded-circle bg-primary p-2 me-3">
                    <i className="bi bi-speedometer2 text-white fs-4"></i>
                  </div>
                  <h5 className="mb-0" style={{ fontSize: "1.05rem" }}>Dashboard</h5>
                </div>
                <p className="card-text" style={{ fontSize: "0.9rem" }}>
                  Get a comprehensive overview of your business with real-time metrics, 
                  sales activity tracking, and industry health indicators.
                </p>
              </Card.Body>
            </Card>
          </Col>

          <Col md={6}>
            <Card className="h-100 shadow-sm border-success border-top border-4">
              <Card.Body>
                <div className="d-flex align-items-center mb-2">
                  <div className="rounded-circle bg-success p-2 me-3">
                    <i className="bi bi-graph-up-arrow text-white fs-4"></i>
                  </div>
                  <h5 className="mb-0" style={{ fontSize: "1.05rem" }}>Forecast</h5>
                </div>
                <p className="card-text" style={{ fontSize: "0.9rem" }}>
                  Leverage AI-powered predictions to anticipate sales trends, 
                  optimize inventory, and make data-driven business decisions.
                </p>
              </Card.Body>
            </Card>
          </Col>

          <Col md={6}>
            <Card className="h-100 shadow-sm border-info border-top border-4">
              <Card.Body>
                <div className="d-flex align-items-center mb-2">
                  <div className="rounded-circle bg-info p-2 me-3">
                    <i className="bi bi-box-seam text-white fs-4"></i>
                  </div>
                  <h5 className="mb-0" style={{ fontSize: "1.05rem" }}>Inventory</h5>
                </div>
                <p className="card-text" style={{ fontSize: "0.9rem" }}>
                  Efficiently manage your stock with automated low-stock alerts, 
                  detailed item tracking, and streamlined inventory operations.
                </p>
              </Card.Body>
            </Card>
          </Col>

          <Col md={6}>
            <Card className="h-100 shadow-sm border-warning border-top border-4">
              <Card.Body>
                <div className="d-flex align-items-center mb-2">
                  <div className="rounded-circle bg-warning p-2 me-3">
                    <i className="bi bi-cart-check text-white fs-4"></i>
                  </div>
                  <h5 className="mb-0" style={{ fontSize: "1.05rem" }}>Sales</h5>
                </div>
                <p className="card-text" style={{ fontSize: "0.9rem" }}>
                  Track all your transactions, monitor customer purchases, 
                  and analyze sales performance across different time periods.
                </p>
              </Card.Body>
            </Card>
          </Col>

          <Col md={6}>
            <Card className="h-100 shadow-sm border-danger border-top border-4">
              <Card.Body>
                <div className="d-flex align-items-center mb-2">
                  <div className="rounded-circle bg-danger p-2 me-3">
                    <i className="bi bi-file-earmark-text text-white fs-4"></i>
                  </div>
                  <h5 className="mb-0" style={{ fontSize: "1.05rem" }}>Purchase Orders</h5>
                </div>
                <p className="card-text" style={{ fontSize: "0.9rem" }}>
                  Create and manage purchase orders with vendors, 
                  track order status, and automatically update inventory upon approval.
                </p>
              </Card.Body>
            </Card>
          </Col>

          <Col md={6}>
            <Card className="h-100 shadow-sm border-secondary border-top border-4">
              <Card.Body>
                <div className="d-flex align-items-center mb-2">
                  <div className="rounded-circle bg-secondary p-2 me-3">
                    <i className="bi bi-person-circle text-white fs-4"></i>
                  </div>
                  <h5 className="mb-0" style={{ fontSize: "1.05rem" }}>User Profile</h5>
                </div>
                <p className="card-text" style={{ fontSize: "0.9rem" }}>
                  Manage your account settings, update personal information, 
                  and customize your StockSight experience.
                </p>
              </Card.Body>
            </Card>
          </Col>
        </Row>

        <div className="text-center mt-3">
          <p style={{ fontSize: "0.95rem" }}>
            <strong>StockSight</strong> combines powerful analytics with industry data 
            to provide you with actionable insights for your business.
          </p>
        </div>
      </Modal.Body>
      <Modal.Footer className="border-0 justify-content-center">
        <Button 
          variant="primary" 
          size="lg" 
          onClick={onHide}
          className="px-5"
        >
          Get Started
        </Button>
      </Modal.Footer>
    </Modal>
  );
};

export default WelcomePopup;

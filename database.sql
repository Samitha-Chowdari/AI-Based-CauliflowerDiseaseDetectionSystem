-- ============================================
-- LeafGuardAI Database Setup
-- AI-Based Cauliflower Disease Detection System
-- ============================================

CREATE DATABASE IF NOT EXISTS cauliflower;

USE cauliflower;

-- ============================================
-- Users Table
-- ============================================

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL
);

-- ============================================
-- Prediction Results Table
-- ============================================

CREATE TABLE IF NOT EXISTS results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    disease VARCHAR(100) NOT NULL,
    confidence FLOAT,
    date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
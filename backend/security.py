import re
import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel


# === Validation Models ===

class ValidationResult(BaseModel):
    valid: bool = True
    message: str = ""


class InjectionResult(BaseModel):
    detected: bool = False
    reason: str = ""


class OutputValidationResult(BaseModel):
    cleaned_output: str = ""
    warnings: list[str] = []


# === 1. InputValidator ===

class InputValidator:
    MAX_LENGTH = 10000
    BLOCKED_PREFIXES = ["javascript:", "data:", "vbscript:"]

    def validate(self, text: str) -> ValidationResult:
        if not text or not text.strip():
            return ValidationResult(valid=False, message="Message is empty")

        if len(text) > self.MAX_LENGTH:
            return ValidationResult(valid=False, message=f"Message exceeds {self.MAX_LENGTH} characters")

        for prefix in self.BLOCKED_PREFIXES:
            if text.strip().lower().startswith(prefix):
                return ValidationResult(valid=False, message=f"URL scheme '{prefix}' is not allowed")

        return ValidationResult(valid=True)


# === 2. InjectionDetector ===

class InjectionDetector:
    PATTERNS = [
        (r"ignore\s+(all\s+)?(previous|prior)\s+(instructions|directions|prompts?)", "Instruction override attempt"),
        (r"(reveal|show|print|display|output|leak|dump)\s+(your|the|its)\s+(system|internal|hidden)\s+(prompt|instructions|prompt)", "System prompt extraction attempt"),
        (r"you\s+are\s+now\s+(DAN|jailbroken|unrestricted|freed)", "Jailbreak attempt (DAN)"),
        (r"developer\s+mode\s*(enabled|activated|on)", "Developer mode attempt"),
        (r"bypass\s+(all\s+)?(restrictions|safeguards|filters|limitations)", "Bypass attempt"),
        (r"act\s+as\s+(if\s+)?(you\s+are\s+)?(DAN|GPT-\d|an?\s+AI\s+without)", "Impersonation attempt"),
        (r"(new|updated|override)\s+(instructions|rules|prompt)\s*:", "Instruction override attempt"),
        (r"---\s*(end|start)\s*(of)?\s*(prompt|instructions)", "Delimiter injection"),
        (r"pretend\s+(you\s+are|to\s+be)\s+(an?\s+)?(unrestricted|unfiltered|uncensored)", "Pretend mode attempt"),
        (r"forget\s+(all\s+)?(previous|prior)\s+(instructions|context|conversation)", "Memory override attempt"),
    ]

    def __init__(self):
        self.compiled = [(re.compile(p, re.IGNORECASE), reason) for p, reason in self.PATTERNS]

    def check(self, text: str) -> InjectionResult:
        for pattern, reason in self.compiled:
            if pattern.search(text):
                return InjectionResult(detected=True, reason=reason)
        return InjectionResult(detected=False)


# === 3. PIIDetector ===

class PIIDetector:
    PATTERNS: dict[str, re.Pattern] = {
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
    }

    MASK_MAP: dict[str, str] = {
        "email": "[EMAIL REDACTED]",
        "phone": "[PHONE REDACTED]",
        "ssn": "[SSN REDACTED]",
        "credit_card": "[CARD REDACTED]",
    }

    def detect(self, text: str) -> dict[str, list[str]]:
        found: dict[str, list[str]] = {}
        for pii_type, pattern in self.PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                found[pii_type] = matches
        return found

    def mask(self, text: str) -> str:
        masked = text
        for pii_type, pattern in self.PATTERNS.items():
            masked = pattern.sub(self.MASK_MAP[pii_type], masked)
        return masked


# === 4. OutputValidator ===

class OutputValidator:
    SECRET_PATTERNS = [
        re.compile(r"(?i)(api[_\s]?key|secret|token|password)\s*[:=]\s*\S+"),
        re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
        re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    ]

    def __init__(self):
        self.pii = PIIDetector()

    def validate(self, output: str) -> OutputValidationResult:
        warnings = []

        for pattern in self.SECRET_PATTERNS:
            if pattern.search(output):
                output = pattern.sub("[SECRET REDACTED]", output)
                warnings.append("Potential secret key redacted from output")

        if self.pii.detect(output):
            output = self.pii.mask(output)
            warnings.append("PII redacted from output")

        return OutputValidationResult(cleaned_output=output, warnings=warnings)


# === 5. SecurityPipeline ===

class SecurityPipeline:
    def __init__(self):
        self.input_validator = InputValidator()
        self.injection_detector = InjectionDetector()
        self.pii_detector = PIIDetector()
        self.output_validator = OutputValidator()

    def validate_input(self, text: str) -> tuple[bool, str, list[str]]:
        notes = []

        vr = self.input_validator.validate(text)
        if not vr.valid:
            return False, "", [vr.message]

        ir = self.injection_detector.check(text)
        if ir.detected:
            return False, "", [f"Blocked: {ir.reason}"]

        pii_found = self.pii_detector.detect(text)
        if pii_found:
            notes.append(f"PII detected in input: {list(pii_found.keys())}")

        return True, text, notes

    def validate_output(self, text: str) -> OutputValidationResult:
        return self.output_validator.validate(text)


# Singleton
security_pipeline = SecurityPipeline()


# === Metrics Collector ===

class MetricsCollector:
    def __init__(self):
        self.total = 0
        self.successful = 0
        self.failed = 0
        self.blocked = 0
        self._latency_sum = 0.0
        self._latency_count = 0
        self._by_intent: dict[str, int] = {
            "conversation": 0,
            "knowledge": 0,
            "coding": 0,
            "escalation": 0,
        }

    def record(self, latency_ms: float, success: bool, blocked: bool = False, intent: str = ""):
        self.total += 1
        if blocked:
            self.blocked += 1
        elif success:
            self.successful += 1
        else:
            self.failed += 1
        self._latency_sum += latency_ms
        self._latency_count += 1
        if intent in self._by_intent:
            self._by_intent[intent] += 1

    @property
    def summary(self) -> dict:
        avg_latency = round(self._latency_sum / self._latency_count, 2) if self._latency_count > 0 else 0.0
        return {
            "total_requests": self.total,
            "successful": self.successful,
            "failed": self.failed,
            "blocked": self.blocked,
            "average_latency_ms": avg_latency,
            "conversation_requests": self._by_intent["conversation"],
            "knowledge_requests": self._by_intent["knowledge"],
            "coding_requests": self._by_intent["coding"],
            "escalation_requests": self._by_intent["escalation"],
        }


metrics_collector = MetricsCollector()


# === Request ID ===

def generate_request_id() -> str:
    return str(uuid.uuid4())
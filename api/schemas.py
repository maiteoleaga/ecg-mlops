"""Modelos Pydantic para la API."""
from typing import Dict
from pydantic import BaseModel, Field, field_validator

# Longitud esperada de la señal ECG (187 puntos en el MIT-BIH)
EXPECTED_SIGNAL_LENGTH = 187


class PredictRequest(BaseModel):
    """Cuerpo del request a /predict."""

    signal: list[float] = Field(
        ...,
        description=f"Señal ECG de {EXPECTED_SIGNAL_LENGTH} puntos, normalizada en [0, 1]",
        examples=[[0.0] * EXPECTED_SIGNAL_LENGTH],
    )

    @field_validator("signal")
    @classmethod
    def check_length(cls, v: list[float]) -> list[float]:
        if len(v) != EXPECTED_SIGNAL_LENGTH:
            raise ValueError(
                f"La señal debe tener exactamente {EXPECTED_SIGNAL_LENGTH} puntos. "
                f"Recibidos: {len(v)}"
            )
        return v


class PredictResponse(BaseModel):
    """Cuerpo de la respuesta de /predict."""

    predicted_class: int = Field(..., ge=0, le=4, description="Índice de la clase predicha (0-4)")
    class_name: str = Field(..., description="Nombre legible de la clase predicha")
    probabilities: Dict[str, float] = Field(
        ...,
        description="Probabilidad de cada clase (suman 1.0)",
    )


class HealthResponse(BaseModel):
    """Cuerpo de la respuesta de /health."""

    status: str
    model_loaded: bool
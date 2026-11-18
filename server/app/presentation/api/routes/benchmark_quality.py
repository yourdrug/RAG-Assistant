"""Admin Benchmark acceptance: dataset thresholds, composition, noise, human labels, judge agreement."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from application.benchmark import BenchmarkQualityService
from application.ports.rate_limit import RateLimitPolicyName
from domain.benchmark.agreement import HumanLabel
from domain.benchmark.policy import DatasetPolicy
from presentation.api.auth_dependencies import require_admin
from presentation.api.dependencies import create_benchmark_quality_service
from presentation.api.rate_limit import rate_limit
from presentation.api.schemas.auth import CurrentUser
from presentation.api.schemas.benchmark_quality import (
    AcceptanceThresholdsBody,
    CompositionResponse,
    DatasetPoliciesResponse,
    DatasetPolicyResponse,
    HumanLabelBody,
    HumanLabelResponse,
    HumanLabelsResponse,
    JudgeAgreementResponse,
    JudgeCalibrationResponse,
    JudgeCalibrationsResponse,
    NoiseRequest,
    ReachabilityResponse,
)


router = APIRouter(tags=["benchmark-quality"])

Admin = Annotated[CurrentUser, Depends(require_admin)]
Service = Annotated[BenchmarkQualityService, Depends(create_benchmark_quality_service)]
WRITE_LIMIT = [Depends(rate_limit(RateLimitPolicyName.WRITE))]
DATASET_PATH = "/admin/benchmark/datasets/{dataset}"


def policy_response(policy: DatasetPolicy) -> DatasetPolicyResponse:
    return DatasetPolicyResponse(
        dataset=policy.dataset,
        thresholds=AcceptanceThresholdsBody(**policy.thresholds.to_dict()),
        noise=policy.noise.to_dict() if policy.noise is not None else None,
        calibration=policy.calibration.to_dict() if policy.calibration is not None else None,
        updated_by=policy.updated_by,
        update_date=policy.update_date,
    )


def label_response(label: HumanLabel) -> HumanLabelResponse:
    return HumanLabelResponse(
        id=label.id,
        run_id=label.run_id,
        question_key=label.question_key,
        metric=label.metric.value,
        score=label.score,
        labeler_id=label.labeler_id,
        note=label.note,
        update_date=label.update_date,
    )


@router.get("/admin/benchmark/policies")
async def list_policies(*, _admin: Admin, service: Service) -> DatasetPoliciesResponse:
    return DatasetPoliciesResponse(policies=[policy_response(p) for p in await service.list_policies()])


@router.get(f"{DATASET_PATH}/policy")
async def get_policy(*, dataset: str, _admin: Admin, service: Service) -> DatasetPolicyResponse:
    return policy_response(await service.get_policy(dataset))


@router.put(f"{DATASET_PATH}/policy/thresholds", dependencies=WRITE_LIMIT)
async def update_thresholds(
    *, dataset: str, body: AcceptanceThresholdsBody, admin: Admin, service: Service
) -> DatasetPolicyResponse:
    policy = await service.update_thresholds(dataset, body.model_dump(), admin.id)
    return policy_response(policy)


@router.post(f"{DATASET_PATH}/policy/noise", dependencies=WRITE_LIMIT)
async def measure_noise(
    *, dataset: str, body: NoiseRequest, admin: Admin, service: Service
) -> DatasetPolicyResponse:
    """Standard deviation from finished repeated runs of one configuration; launches nothing."""
    return policy_response(await service.measure_noise(dataset, body.run_ids, admin.id))


@router.get(f"{DATASET_PATH}/calibrations")
async def list_calibrations(*, dataset: str, _admin: Admin, service: Service) -> JudgeCalibrationsResponse:
    """Saved judge calibrations of a dataset, newest first."""
    return JudgeCalibrationsResponse(
        dataset=dataset,
        calibrations=[
            JudgeCalibrationResponse(
                id=item.id,
                run_id=item.run_id,
                judge_model=item.judge.model,
                judge_rubric_version=item.judge.rubric_version,
                metrics=item.metrics,
                measured_at=item.measured_at,
                created_by=item.created_by,
            )
            for item in await service.list_calibrations(dataset)
        ],
    )


@router.get(f"{DATASET_PATH}/reachability")
async def reachability(
    *,
    dataset: str,
    space_id: Annotated[int | None, Query(ge=1)] = None,
    _admin: Admin,
    service: Service,
) -> ReachabilityResponse:
    """Expected fragments found whole inside one chunk of the space; calls no model."""
    space, report = await service.fragment_reachability(dataset, space_id)
    return ReachabilityResponse(
        dataset=dataset, space_id=space.id, pack_key=space.pack_key, **report.to_dict()
    )


@router.get(f"{DATASET_PATH}/composition")
async def composition(*, dataset: str, _admin: Admin, service: Service) -> CompositionResponse:
    report = await service.composition(dataset)
    return CompositionResponse(dataset=dataset, **report.to_dict())


@router.get("/admin/benchmark/runs/{run_id:int}/labels")
async def list_labels(*, run_id: int, _admin: Admin, service: Service) -> HumanLabelsResponse:
    return HumanLabelsResponse(labels=[label_response(label) for label in await service.list_labels(run_id)])


@router.put("/admin/benchmark/runs/{run_id:int}/labels", dependencies=WRITE_LIMIT)
async def save_label(
    *, run_id: int, body: HumanLabelBody, admin: Admin, service: Service
) -> HumanLabelResponse:
    label = await service.save_label(
        run_id=run_id,
        question_key=body.question_key,
        metric=body.metric,
        score=body.score,
        note=body.note,
        user_id=admin.id,
    )
    return label_response(label)


@router.delete("/admin/benchmark/runs/{run_id:int}/labels", status_code=204, dependencies=WRITE_LIMIT)
async def delete_label(
    *,
    run_id: int,
    question_key: Annotated[str, Query(min_length=1, max_length=200)],
    metric: Annotated[Literal["correctness", "faithfulness"], Query()],
    admin: Admin,
    service: Service,
) -> None:
    if not await service.delete_label(
        run_id=run_id, question_key=question_key, metric=metric, user_id=admin.id
    ):
        raise HTTPException(status_code=404, detail="Label not found")


@router.get("/admin/benchmark/runs/{run_id:int}/agreement")
async def agreement(*, run_id: int, _admin: Admin, service: Service) -> JudgeAgreementResponse:
    return JudgeAgreementResponse(**await service.agreement(run_id))


@router.post("/admin/benchmark/runs/{run_id:int}/calibration", dependencies=WRITE_LIMIT)
async def save_calibration(*, run_id: int, admin: Admin, service: Service) -> DatasetPolicyResponse:
    """Store this run's judge agreement as the calibration of its dataset."""
    return policy_response(await service.save_calibration(run_id, admin.id))

"""
Enhanced Search Monitoring and Health Checks

This module provides comprehensive monitoring, health checks, and system
status reporting for the enhanced vector search system.
"""

from typing import Dict, Any, List
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from genericsuite_codegen.utilities.app_logger import (
    log_error,
)

from .enhanced_search_logging import (
    get_performance_monitor,
    get_enhanced_search_logger
)
from .enhanced_search_error_handler import get_recovery_stats


DEBUG = False


class HealthStatus(Enum):
    """Health status levels for system components."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status for a system component."""
    component_name: str
    status: HealthStatus
    message: str
    last_check: datetime
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component_name": self.component_name,
            "status": self.status.value,
            "message": self.message,
            "last_check": self.last_check.isoformat(),
            "metrics": self.metrics
        }


@dataclass
class SystemHealth:
    """Overall system health status."""
    overall_status: HealthStatus
    components: List[ComponentHealth]
    timestamp: datetime
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_status": self.overall_status.value,
            "components": [comp.to_dict() for comp in self.components],
            "timestamp": self.timestamp.isoformat(),
            "summary": self.summary
        }


class EnhancedSearchMonitor:
    """Comprehensive monitoring for enhanced search system."""

    def __init__(self):
        """Initialize the monitor."""
        self.performance_monitor = get_performance_monitor()
        self.search_logger = get_enhanced_search_logger()
        self.recovery_stats = get_recovery_stats()

        # Health check thresholds
        self.thresholds = {
            "max_error_rate": 0.1,  # 10% error rate threshold
            "max_avg_response_time": 5.0,  # 5 seconds average response time
            "min_success_rate": 0.9,  # 90% success rate minimum
            "max_memory_usage_mb": 500,  # 500MB memory usage threshold
            "max_recovery_failure_rate": 0.2  # 20% recovery failure rate
        }

        # Component health checkers
        self.health_checkers = {
            "dual_search": self._check_dual_search_health,
            "context_determination": self._check_context_determination_health,
            "document_retrieval": self._check_document_retrieval_health,
            "template_management": self._check_template_management_health,
            "error_recovery": self._check_error_recovery_health,
            "performance": self._check_performance_health
        }

    def get_system_health(self) -> SystemHealth:
        """Get comprehensive system health status."""
        components = []
        overall_status = HealthStatus.HEALTHY

        # Check each component
        for component_name, checker in self.health_checkers.items():
            try:
                component_health = checker()
                components.append(component_health)

                # Update overall status based on component status
                if component_health.status == HealthStatus.CRITICAL:
                    overall_status = HealthStatus.CRITICAL
                elif (component_health.status == HealthStatus.WARNING and
                      overall_status != HealthStatus.CRITICAL):
                    overall_status = HealthStatus.WARNING

            except Exception as e:
                log_error(f"Health check failed for {component_name}: {e}")
                components.append(ComponentHealth(
                    component_name=component_name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Health check failed: {e}",
                    last_check=datetime.now(),
                    metrics={}
                ))
                if overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.WARNING

        # Create summary
        summary = self._create_health_summary(components)

        return SystemHealth(
            overall_status=overall_status,
            components=components,
            timestamp=datetime.now(),
            summary=summary
        )

    def _check_dual_search_health(self) -> ComponentHealth:
        """Check dual search component health."""
        stats = self.performance_monitor.get_operation_stats("dual_search")

        if not stats or stats.get("count", 0) == 0:
            return ComponentHealth(
                component_name="dual_search",
                status=HealthStatus.UNKNOWN,
                message="No dual search operations recorded",
                last_check=datetime.now(),
                metrics={}
            )

        error_rate = stats.get("error_count", 0) / stats.get("count", 1)
        avg_duration = stats.get("avg_duration", 0)

        status = HealthStatus.HEALTHY
        messages = []

        if error_rate > self.thresholds["max_error_rate"]:
            status = HealthStatus.CRITICAL
            messages.append(f"High error rate: {error_rate:.2%}")

        if avg_duration > self.thresholds["max_avg_response_time"]:
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.WARNING
            messages.append(f"Slow response time: {avg_duration:.2f}s")

        message = "; ".join(messages) if messages else "Operating normally"

        return ComponentHealth(
            component_name="dual_search",
            status=status,
            message=message,
            last_check=datetime.now(),
            metrics={
                "total_operations": stats.get("count", 0),
                "error_rate": error_rate,
                "avg_duration": avg_duration,
                "success_rate": (
                    stats.get("success_count", 0) / stats.get("count", 1)
                )
            }
        )

    def _check_context_determination_health(self) -> ComponentHealth:
        """Check context determination component health."""
        stats = self.performance_monitor.get_operation_stats(
            "context_determination"
        )

        if not stats or stats.get("count", 0) == 0:
            return ComponentHealth(
                component_name="context_determination",
                status=HealthStatus.UNKNOWN,
                message="No context determination operations recorded",
                last_check=datetime.now(),
                metrics={}
            )

        error_rate = stats.get("error_count", 0) / stats.get("count", 1)
        avg_duration = stats.get("avg_duration", 0)

        status = HealthStatus.HEALTHY
        messages = []

        if error_rate > self.thresholds["max_error_rate"]:
            # Context determination errors are less critical
            status = HealthStatus.WARNING
            messages.append(f"Error rate: {error_rate:.2%}")

        if avg_duration > 2.0:  # Context determination should be fast
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.WARNING
            messages.append(
                f"Slow context determination: {avg_duration:.2f}s"
            )

        message = "; ".join(messages) if messages else "Operating normally"

        return ComponentHealth(
            component_name="context_determination",
            status=status,
            message=message,
            last_check=datetime.now(),
            metrics={
                "total_operations": stats.get("count", 0),
                "error_rate": error_rate,
                "avg_duration": avg_duration
            }
        )

    def _check_document_retrieval_health(self) -> ComponentHealth:
        """Check document retrieval component health."""
        stats = self.performance_monitor.get_operation_stats(
            "document_retrieval")

        if not stats or stats.get("count", 0) == 0:
            return ComponentHealth(
                component_name="document_retrieval",
                status=HealthStatus.HEALTHY,
                message="No document retrieval operations recorded",
                last_check=datetime.now(),
                metrics={}
            )

        error_rate = stats.get("error_count", 0) / stats.get("count", 1)
        avg_duration = stats.get("avg_duration", 0)

        status = HealthStatus.HEALTHY
        messages = []

        # Higher threshold for document retrieval
        if error_rate > 0.2:
            status = HealthStatus.WARNING
            messages.append(f"Error rate: {error_rate:.2%}")

        # Document retrieval can be slower
        if avg_duration > 3.0:
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.WARNING
            messages.append(
                f"Slow document retrieval: {avg_duration:.2f}s"
            )

        message = "; ".join(messages) if messages else "Operating normally"

        return ComponentHealth(
            component_name="document_retrieval",
            status=status,
            message=message,
            last_check=datetime.now(),
            metrics={
                "total_operations": stats.get("count", 0),
                "error_rate": error_rate,
                "avg_duration": avg_duration
            }
        )

    def _check_template_management_health(self) -> ComponentHealth:
        """Check template management component health."""
        stats = self.performance_monitor.get_operation_stats("template_load")

        status = HealthStatus.HEALTHY
        message = "Template management operating normally"
        metrics = {}

        if stats and stats.get("count", 0) > 0:
            error_rate = stats.get("error_count", 0) / stats.get("count", 1)
            metrics = {
                "template_loads": stats.get("count", 0),
                "error_rate": error_rate,
                "avg_load_time": stats.get("avg_duration", 0)
            }

            # Template loading should be very reliable
            if error_rate > 0.1:
                status = HealthStatus.WARNING
                message = f"Template loading errors: {error_rate:.2%}"

        return ComponentHealth(
            component_name="template_management",
            status=status,
            message=message,
            last_check=datetime.now(),
            metrics=metrics
        )

    def _check_error_recovery_health(self) -> ComponentHealth:
        """Check error recovery system health."""
        recovery_stats = self.recovery_stats.get_stats()

        if recovery_stats["total_attempts"] == 0:
            return ComponentHealth(
                component_name="error_recovery",
                status=HealthStatus.HEALTHY,
                message="No error recovery attempts recorded",
                last_check=datetime.now(),
                metrics=recovery_stats
            )

        recovery_rate = recovery_stats["recovery_rate"]

        status = HealthStatus.HEALTHY
        message = f"Recovery rate: {recovery_rate:.2%}"

        # 80% recovery rate threshold
        if recovery_rate < 0.8:
            status = HealthStatus.WARNING
            message = f"Low recovery rate: {recovery_rate:.2%}"

        # 50% recovery rate critical threshold
        if recovery_rate < 0.5:
            status = HealthStatus.CRITICAL
            message = f"Critical recovery rate: {recovery_rate:.2%}"

        return ComponentHealth(
            component_name="error_recovery",
            status=status,
            message=message,
            last_check=datetime.now(),
            metrics=recovery_stats
        )

    def _check_performance_health(self) -> ComponentHealth:
        """Check overall performance health."""
        all_stats = self.performance_monitor.get_operation_stats()

        if not all_stats:
            return ComponentHealth(
                component_name="performance",
                status=HealthStatus.UNKNOWN,
                message="No performance data available",
                last_check=datetime.now(),
                metrics={}
            )

        # Calculate overall metrics
        total_operations = sum(stats.get("count", 0)
                               for stats in all_stats.values())
        total_errors = sum(stats.get("error_count", 0)
                           for stats in all_stats.values())

        overall_error_rate = total_errors / \
            total_operations if total_operations > 0 else 0

        # Calculate weighted average duration
        total_duration = sum(
            stats.get("total_duration", 0) for stats in all_stats.values()
        )
        avg_duration = (
            total_duration / total_operations if total_operations > 0 else 0
        )

        status = HealthStatus.HEALTHY
        messages = []

        if overall_error_rate > self.thresholds["max_error_rate"]:
            status = HealthStatus.WARNING
            messages.append(
                f"High overall error rate: {overall_error_rate:.2%}")

        if avg_duration > self.thresholds["max_avg_response_time"]:
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.WARNING
            messages.append(f"Slow overall response time: {avg_duration:.2f}s")

        message = "; ".join(
            messages) if messages else "Performance within normal ranges"

        return ComponentHealth(
            component_name="performance",
            status=status,
            message=message,
            last_check=datetime.now(),
            metrics={
                "total_operations": total_operations,
                "overall_error_rate": overall_error_rate,
                "avg_duration": avg_duration,
                "operations_by_type": {
                    op_type: stats.get("count", 0)
                    for op_type, stats in all_stats.items()
                }
            }
        )

    def _create_health_summary(
        self, components: List[ComponentHealth]
    ) -> Dict[str, Any]:
        """Create health summary from component health checks."""
        status_counts = {status.value: 0 for status in HealthStatus}

        for component in components:
            status_counts[component.status.value] += 1

        # Get recent error summary
        error_summary = self.search_logger.get_error_summary(hours=1)

        return {
            "total_components": len(components),
            "status_distribution": status_counts,
            "recent_errors": error_summary["total_errors"],
            "error_types": list(error_summary["error_counts"].keys()),
            "system_uptime": self._get_system_uptime(),
            "last_health_check": datetime.now().isoformat()
        }

    def _get_system_uptime(self) -> str:
        """Get system uptime (simplified - enhance with actual uptime)."""
        # This is a simplified implementation
        # In a real system, you'd track actual startup time
        return "Unknown"

    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        all_stats = self.performance_monitor.get_operation_stats()
        recent_metrics = self.performance_monitor.get_recent_metrics(
            limit=1000)
        error_summary = self.search_logger.get_error_summary(hours=hours)

        return {
            "report_period_hours": hours,
            "timestamp": datetime.now().isoformat(),
            "operation_statistics": all_stats,
            "recent_metrics_count": len(recent_metrics),
            "error_summary": error_summary,
            "recovery_statistics": self.recovery_stats.get_stats(),
            "performance_thresholds": self.thresholds
        }

    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        system_health = self.get_system_health()
        performance_report = self.get_performance_report(hours=1)

        return {
            "system_health": system_health.to_dict(),
            "performance_summary": {
                "total_operations": sum(
                    stats.get("count", 0)
                    for stats in performance_report[
                        "operation_statistics"
                    ].values()
                ),
                "recent_errors": performance_report[
                    "error_summary"
                ]["total_errors"],
                "recovery_rate": performance_report[
                    "recovery_statistics"
                ]["recovery_rate"]
            },
            "alerts": self._get_active_alerts(system_health),
            "timestamp": datetime.now().isoformat()
        }

    def _get_active_alerts(
        self, system_health: SystemHealth
    ) -> List[Dict[str, Any]]:
        """Get active alerts based on system health."""
        alerts = []

        for component in system_health.components:
            if component.status in [
                HealthStatus.WARNING, HealthStatus.CRITICAL
            ]:
                alerts.append({
                    "component": component.component_name,
                    "severity": component.status.value,
                    "message": component.message,
                    "timestamp": component.last_check.isoformat()
                })

        return alerts


# Global monitor instance
_global_monitor = EnhancedSearchMonitor()


def get_monitor() -> EnhancedSearchMonitor:
    """Get the global monitor instance."""
    return _global_monitor


def get_system_health() -> SystemHealth:
    """Get current system health status."""
    return _global_monitor.get_system_health()


def get_performance_report(hours: int = 24) -> Dict[str, Any]:
    """Get performance report for specified time period."""
    return _global_monitor.get_performance_report(hours)


def get_dashboard_data() -> Dict[str, Any]:
    """Get monitoring dashboard data."""
    return _global_monitor.get_monitoring_dashboard_data()

"""
Unit tests for Dash page imports.

Tests verify that all page modules can be imported without errors,
catching issues like invalid component names (e.g., html.Style).
"""

import pytest
import sys
import os
import ast

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestComponentImports:
    """Tests to verify all components import correctly."""

    def test_import_navbar_component(self):
        """Test navbar component imports without errors."""
        from app.components import navbar
        assert hasattr(navbar, 'create_navbar')

    def test_import_charts_component(self):
        """Test charts component imports without errors."""
        from app.components import charts
        assert hasattr(charts, 'create_pnl_chart')

    def test_import_tables_component(self):
        """Test tables component imports without errors."""
        from app.components import tables
        assert hasattr(tables, 'metric_card')

    def test_import_monitoring_component(self):
        """Test monitoring component imports without errors."""
        from app.components import monitoring
        assert hasattr(monitoring, 'create_monitoring_dashboard')

    def test_import_connection_status_component(self):
        """Test connection_status component imports without errors."""
        from app.components import connection_status
        assert hasattr(connection_status, 'connection_badge')


class TestDashHtmlComponents:
    """Tests to verify valid Dash html component usage."""

    def test_html_style_does_not_exist(self):
        """Verify html.Style doesn't exist (common mistake that causes runtime errors)."""
        from dash import html
        
        # html.Style is a common mistake - it should not exist
        # Using it causes: AttributeError: module 'dash.html' has no attribute 'Style'
        assert not hasattr(html, 'Style'), (
            "html.Style should not exist - "
            "use dcc.Markdown with dangerously_allow_html=True for CSS injection instead"
        )

    def test_dcc_module_has_markdown(self):
        """Verify dcc.Markdown exists for CSS injection workaround."""
        from dash import dcc
        
        assert hasattr(dcc, 'Markdown'), "dcc.Markdown should exist for CSS injection"

    def test_common_html_components_exist(self):
        """Verify commonly used html components exist."""
        from dash import html
        
        common_components = [
            'Div', 'Span', 'P', 'A', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6',
            'Ul', 'Ol', 'Li', 'Table', 'Tr', 'Td', 'Th',
            'Button', 'Img', 'Br', 'Hr', 'Small', 'Strong', 'Em', 'I', 'B'
        ]
        
        for component in common_components:
            assert hasattr(html, component), f"html.{component} should exist"

    def test_use_dcc_input_not_html_input(self):
        """Verify dcc.Input exists (html.Input does not exist)."""
        from dash import dcc
        
        # html.Input doesn't exist, use dcc.Input instead
        assert hasattr(dcc, 'Input'), "Use dcc.Input instead of html.Input"


class TestServicesImport:
    """Tests to verify services module imports correctly."""

    def test_import_services(self):
        """Test services module imports without errors."""
        from app import services
        assert hasattr(services, 'get_services')


class TestHtmlStyleStaticCheck:
    """Static analysis to detect html.Style usage in source files."""

    def _find_html_style_usage(self, file_path):
        """Parse Python file and find html.Style usage."""
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Simple string search for the problematic pattern
        if 'html.Style' in content:
            return True
        return False

    def test_no_html_style_in_pages(self):
        """Verify no page files use html.Style."""
        pages_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'app', 'pages'
        )
        
        problematic_files = []
        for filename in os.listdir(pages_dir):
            if filename.endswith('.py'):
                file_path = os.path.join(pages_dir, filename)
                if self._find_html_style_usage(file_path):
                    problematic_files.append(filename)
        
        assert len(problematic_files) == 0, (
            f"Found html.Style usage in: {problematic_files}. "
            "Use dcc.Markdown with dangerously_allow_html=True instead."
        )

    def test_no_html_style_in_components(self):
        """Verify no component files use html.Style."""
        components_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'app', 'components'
        )
        
        problematic_files = []
        for filename in os.listdir(components_dir):
            if filename.endswith('.py'):
                file_path = os.path.join(components_dir, filename)
                if self._find_html_style_usage(file_path):
                    problematic_files.append(filename)
        
        assert len(problematic_files) == 0, (
            f"Found html.Style usage in: {problematic_files}. "
            "Use dcc.Markdown with dangerously_allow_html=True instead."
        )
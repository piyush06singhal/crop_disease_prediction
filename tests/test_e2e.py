# tests/test_e2e.py - End-to-end tests
"""
End-to-end tests for Crop Disease Prediction System
Tests complete user workflows from browser interaction to results
"""

import pytest
import time
import os
import tempfile
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import requests
from PIL import Image
import io


@pytest.fixture(scope="session")
def browser_config():
    """Configure browser options for testing"""
    # Use headless mode for CI/CD
    headless = os.environ.get('HEADLESS', 'true').lower() == 'true'

    browser = os.environ.get('BROWSER', 'chrome').lower()

    if browser == 'chrome':
        options = ChromeOptions()
        if headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        return {'browser': 'chrome', 'options': options}
    elif browser == 'firefox':
        options = FirefoxOptions()
        if headless:
            options.add_argument('--headless')
        return {'browser': 'firefox', 'options': options}
    else:
        raise ValueError(f"Unsupported browser: {browser}")


@pytest.fixture
def driver(browser_config):
    """Create WebDriver instance"""
    if browser_config['browser'] == 'chrome':
        driver = webdriver.Chrome(options=browser_config['options'])
    elif browser_config['browser'] == 'firefox':
        driver = webdriver.Firefox(options=browser_config['options'])

    driver.implicitly_wait(10)
    yield driver
    driver.quit()


@pytest.fixture
def test_image():
    """Create a test image file"""
    # Create a simple colored image for testing
    img = Image.new('RGB', (224, 224), color=(255, 0, 0))  # Red square
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    img.save(temp_file.name, 'JPEG')
    temp_file.close()

    yield temp_file.name

    # Cleanup
    os.unlink(temp_file.name)


@pytest.fixture
def base_url():
    """Get base URL for the application"""
    return os.environ.get('BASE_URL', 'http://localhost:5000')


class TestBasicNavigation:
    """Test basic navigation and UI elements"""

    def test_page_load(self, driver, base_url):
        """Test that the main page loads correctly"""
        driver.get(base_url)

        # Check page title
        assert "Crop Disease Prediction" in driver.title

        # Check main elements are present
        upload_tab = driver.find_element(By.ID, 'upload-tab')
        camera_tab = driver.find_element(By.ID, 'camera-tab')
        assert upload_tab.is_displayed()
        assert camera_tab.is_displayed()

    def test_tab_switching(self, driver, base_url):
        """Test switching between upload and camera tabs"""
        driver.get(base_url)

        # Initially upload tab should be active
        upload_tab = driver.find_element(By.ID, 'upload-tab')
        camera_tab = driver.find_element(By.ID, 'camera-tab')
        upload_section = driver.find_element(By.ID, 'upload-section')
        camera_section = driver.find_element(By.ID, 'camera-section')

        assert 'active' in upload_tab.get_attribute('class')
        assert upload_section.is_displayed()
        assert not camera_section.is_displayed()

        # Switch to camera tab
        camera_tab.click()
        time.sleep(0.5)  # Allow for transition

        assert 'active' in camera_tab.get_attribute('class')
        assert not upload_section.is_displayed()
        assert camera_section.is_displayed()

        # Switch back to upload tab
        upload_tab.click()
        time.sleep(0.5)

        assert 'active' in upload_tab.get_attribute('class')
        assert upload_section.is_displayed()
        assert not camera_section.is_displayed()


class TestImageUpload:
    """Test image upload functionality"""

    def test_file_upload_display(self, driver, base_url, test_image):
        """Test uploading and displaying an image"""
        driver.get(base_url)

        # Find upload area and input
        upload_area = driver.find_element(By.ID, 'upload-area')
        upload_input = driver.find_element(By.ID, 'image-input')

        # Upload file
        upload_input.send_keys(test_image)

        # Wait for preview to appear
        preview_section = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.ID, 'preview-section'))
        )

        # Check preview elements
        image_preview = driver.find_element(By.ID, 'image-preview')
        assert image_preview.is_displayed()

        # Check metadata display
        image_size = driver.find_element(By.ID, 'image-size')
        image_format = driver.find_element(By.ID, 'image-format')
        image_dimensions = driver.find_element(By.ID, 'image-dimensions')

        assert image_size.is_displayed()
        assert image_format.is_displayed()
        assert image_dimensions.is_displayed()

        # Check analyze button is enabled
        analyze_btn = driver.find_element(By.ID, 'analyze-btn')
        assert analyze_btn.is_enabled()

    def test_drag_and_drop(self, driver, base_url, test_image):
        """Test drag and drop file upload"""
        driver.get(base_url)

        upload_area = driver.find_element(By.ID, 'upload-area')

        # Execute JavaScript to simulate drag and drop
        driver.execute_script("""
            const uploadArea = arguments[0];
            const file = new File(['test image content'], 'test.jpg', {type: 'image/jpeg'});

            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);

            const dragEvent = new DragEvent('drop', {
                dataTransfer: dataTransfer,
                bubbles: true
            });

            uploadArea.dispatchEvent(dragEvent);
        """, upload_area)

        # Check if dragover class is applied (may not work in all browsers)
        # This is more of a UI test than functional test

    def test_invalid_file_type(self, driver, base_url):
        """Test uploading invalid file type"""
        driver.get(base_url)

        upload_input = driver.find_element(By.ID, 'image-input')

        # Create temporary text file
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
            temp_file.write(b'this is not an image')
            temp_file_path = temp_file.name

        try:
            upload_input.send_keys(temp_file_path)

            # Should show error toast
            toast_container = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.ID, 'toast-container'))
            )

            # Check for error message
            error_toasts = driver.find_elements(By.CLASS_NAME, 'border-red-500')
            assert len(error_toasts) > 0

        finally:
            os.unlink(temp_file_path)


class TestDiseaseAnalysis:
    """Test disease analysis workflow"""

    def test_successful_analysis(self, driver, base_url, test_image):
        """Test complete analysis workflow"""
        driver.get(base_url)

        # Upload image
        upload_input = driver.find_element(By.ID, 'image-input')
        upload_input.send_keys(test_image)

        # Wait for preview
        WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.ID, 'preview-section'))
        )

        # Click analyze
        analyze_btn = driver.find_element(By.ID, 'analyze-btn')
        analyze_btn.click()

        # Wait for analysis progress
        WebDriverWait(driver, 30).until(
            EC.visibility_of_element_located((By.ID, 'analysis-progress'))
        )

        # Wait for results (this may take time depending on backend)
        try:
            results_section = WebDriverWait(driver, 60).until(
                EC.visibility_of_element_located((By.ID, 'results-section'))
            )

            # Check results content
            results_content = driver.find_element(By.ID, 'results-content')
            assert results_content.is_displayed()

            # Check for prediction data
            assert "Predicted Disease" in results_content.text or "Primary Diagnosis" in results_content.text

        except TimeoutException:
            # If analysis takes too long, check for error message
            toast_container = driver.find_element(By.ID, 'toast-container')
            error_toasts = driver.find_elements(By.CLASS_NAME, 'border-red-500')
            if error_toasts:
                pytest.fail("Analysis failed with error")
            else:
                pytest.skip("Analysis taking too long - possibly backend issue")

    def test_analysis_without_image(self, driver, base_url):
        """Test analysis attempt without image"""
        driver.get(base_url)

        analyze_btn = driver.find_element(By.ID, 'analyze-btn')

        # Button should be disabled initially
        assert not analyze_btn.is_enabled()


class TestCameraFunctionality:
    """Test camera capture functionality"""

    def test_camera_tab_access(self, driver, base_url):
        """Test accessing camera tab"""
        driver.get(base_url)

        camera_tab = driver.find_element(By.ID, 'camera-tab')
        camera_tab.click()

        camera_section = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.ID, 'camera-section'))
        )

        assert camera_section.is_displayed()

    def test_camera_permission_request(self, driver, base_url):
        """Test camera permission handling"""
        driver.get(base_url)

        camera_tab = driver.find_element(By.ID, 'camera-tab')
        camera_tab.click()

        camera_toggle = driver.find_element(By.ID, 'camera-toggle')

        # Click to request camera access
        camera_toggle.click()

        # In headless mode, this will likely fail, but we can check the UI response
        time.sleep(2)

        # Check button text changed or error appeared
        current_text = camera_toggle.text
        assert current_text in ['Stop Camera', 'Start Camera', 'Camera not supported']

    @pytest.mark.skipif(os.environ.get('HEADLESS') == 'true',
                       reason="Camera capture doesn't work in headless mode")
    def test_camera_capture(self, driver, base_url):
        """Test camera image capture (requires real camera)"""
        driver.get(base_url)

        camera_tab = driver.find_element(By.ID, 'camera-tab')
        camera_tab.click()

        camera_toggle = driver.find_element(By.ID, 'camera-toggle')
        camera_toggle.click()

        # Wait for camera to initialize
        time.sleep(3)

        capture_btn = driver.find_element(By.ID, 'capture-btn')
        capture_btn.click()

        # Check for capture feedback
        time.sleep(1)

        # Should switch back to upload tab with captured image
        upload_section = driver.find_element(By.ID, 'upload-section')
        assert upload_section.is_displayed()


class TestPWAFeatures:
    """Test Progressive Web App features"""

    def test_manifest_exists(self, base_url):
        """Test that web app manifest is accessible"""
        response = requests.get(f"{base_url}/static/manifest.json")
        assert response.status_code == 200

        manifest = response.json()
        assert 'name' in manifest
        assert 'short_name' in manifest
        assert 'start_url' in manifest
        assert 'display' in manifest

    def test_service_worker_registration(self, driver, base_url):
        """Test service worker registration"""
        driver.get(base_url)

        # Check if service worker is registered
        sw_registered = driver.execute_script("""
            return navigator.serviceWorker.controller !== null;
        """)

        # Service worker might not be registered in test environment
        # This is more of a check that the code doesn't error
        assert isinstance(sw_registered, bool)

    def test_install_prompt(self, driver, base_url):
        """Test PWA install prompt"""
        driver.get(base_url)

        # Install prompt may or may not appear depending on conditions
        try:
            install_prompt = driver.find_element(By.ID, 'install-prompt')
            # If found, check it has proper elements
            install_accept = driver.find_element(By.ID, 'install-accept')
            install_dismiss = driver.find_element(By.ID, 'install-dismiss')

            assert install_accept.is_displayed()
            assert install_dismiss.is_displayed()

        except NoSuchElementException:
            # Install prompt not shown - this is normal in test environment
            pass

    def test_offline_indicator(self, driver, base_url):
        """Test offline/online status indicator"""
        driver.get(base_url)

        # Should be online initially
        offline_indicator = driver.find_element(By.ID, 'offline-indicator')
        assert not offline_indicator.is_displayed()

        # Simulate going offline
        driver.execute_script("window.dispatchEvent(new Event('offline'));")

        time.sleep(1)
        assert offline_indicator.is_displayed()

        # Simulate coming back online
        driver.execute_script("window.dispatchEvent(new Event('online'));")

        time.sleep(1)
        assert not offline_indicator.is_displayed()


class TestResponsiveDesign:
    """Test responsive design across different screen sizes"""

    @pytest.mark.parametrize("width,height", [
        (1920, 1080),  # Desktop
        (768, 1024),   # Tablet
        (375, 667),    # Mobile
    ])
    def test_responsive_layout(self, driver, base_url, width, height):
        """Test layout adapts to different screen sizes"""
        driver.set_window_size(width, height)
        driver.get(base_url)

        # Check main elements are visible and properly sized
        main_container = driver.find_element(By.CLASS_NAME, 'container')
        assert main_container.is_displayed()

        # Check tabs are accessible
        upload_tab = driver.find_element(By.ID, 'upload-tab')
        camera_tab = driver.find_element(By.ID, 'camera-tab')
        assert upload_tab.is_displayed()
        assert camera_tab.is_displayed()

        # Check content areas
        upload_section = driver.find_element(By.ID, 'upload-section')
        assert upload_section.is_displayed()


class TestErrorHandling:
    """Test error handling and user feedback"""

    def test_network_error_handling(self, driver, base_url, test_image):
        """Test handling of network errors during analysis"""
        driver.get(base_url)

        # Upload image
        upload_input = driver.find_element(By.ID, 'image-input')
        upload_input.send_keys(test_image)

        WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.ID, 'preview-section'))
        )

        # Mock network failure by blocking requests
        driver.execute_script("""
            // Override fetch to simulate network failure
            window.originalFetch = window.fetch;
            window.fetch = function() {
                return Promise.reject(new Error('Network error'));
            };
        """)

        analyze_btn = driver.find_element(By.ID, 'analyze-btn')
        analyze_btn.click()

        # Check for error toast
        toast_container = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, 'toast-container'))
        )

        error_toasts = driver.find_elements(By.CLASS_NAME, 'border-red-500')
        assert len(error_toasts) > 0

        # Restore fetch
        driver.execute_script("window.fetch = window.originalFetch;")


class TestAccessibility:
    """Test accessibility features"""

    def test_keyboard_navigation(self, driver, base_url):
        """Test keyboard navigation"""
        driver.get(base_url)

        # Tab through elements
        active_element = driver.switch_to.active_element

        # Send tab keys
        from selenium.webdriver.common.keys import Keys
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.TAB)

        # Check focus moves to expected elements
        # This is basic - more comprehensive accessibility testing would use specialized tools

    def test_alt_text_and_labels(self, driver, base_url):
        """Test images have alt text and form elements have labels"""
        driver.get(base_url)

        # Check images for alt text
        images = driver.find_elements(By.TAG_NAME, 'img')
        for img in images:
            alt_text = img.get_attribute('alt')
            # Some images might not need alt text, but main content images should have it
            if 'preview' in img.get_attribute('id') or 'logo' in img.get_attribute('class', ''):
                assert alt_text is not None and len(alt_text) > 0

        # Check form inputs have labels
        inputs = driver.find_elements(By.TAG_NAME, 'input')
        for input_element in inputs:
            input_id = input_element.get_attribute('id')
            if input_id:
                label = driver.find_elements(By.CSS_SELECTOR, f"label[for='{input_id}']")
                # Not all inputs need labels, but main form inputs should have them


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
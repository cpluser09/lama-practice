#!/usr/bin/env python3
"""
Test mask upload functionality using Playwright
Tests that uploaded mask files work correctly with both PyTorch and CoreML modes
"""

import asyncio
import tempfile
import os

from playwright.async_api import async_playwright

BASE_URL = "http://localhost:5002"


async def test_mask_upload_pytorch():
    """Test mask upload with PyTorch engine"""

    print("Starting mask upload test with PyTorch engine...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()

        # Navigate to the app
        print(f"Navigating to {BASE_URL}...")
        await page.goto(BASE_URL)
        await page.wait_for_load_state('networkidle')

        # Wait for page to load
        await page.wait_for_selector('#mainCanvas', timeout=5000)

        # Select a test image
        print("Selecting test image...")
        await page.select_option('#testImageSelect', '512')
        await asyncio.sleep(1)

        # Draw a mask and download it first
        print("Drawing initial mask...")
        canvas = await page.query_selector('#mainCanvas')
        box = await canvas.bounding_box()

        if box:
            await page.mouse.move(box['x'] + 100, box['y'] + 100)
            await page.mouse.down()
            await page.mouse.move(box['x'] + 200, box['y'] + 200)
            await page.mouse.up()
            await asyncio.sleep(0.5)

        # Run inpainting to generate mask for download
        print("Running inpainting to download mask...")
        async with page.expect_response('**/inpaint') as response_info:
            await page.click('#inpaintBtn')

        response = await response_info.value
        print(f"Inpaint response status: {response.status}")

        # Wait for download button and download mask
        await page.wait_for_selector('#downloadBtn', state='visible', timeout=60000)

        # Close modal if present
        try:
            modal = await page.query_selector('#compareModal.active')
            if modal:
                await page.keyboard.press('Escape')
                await asyncio.sleep(0.5)
        except:
            pass

        # Set up download handler
        mask_path = None

        async def handle_download(download):
            nonlocal mask_path
            filename = download.suggested_filename
            print(f"Downloaded: {filename}")
            if 'mask' in filename.lower():
                mask_path = await download.path()

        page.on('download', handle_download)

        # Click download
        await page.click('#downloadBtn')
        await asyncio.sleep(2)

        if not mask_path:
            print("FAIL: Could not download mask file")
            await browser.close()
            return False

        print(f"Mask saved to: {mask_path}")

        # Clear the current mask
        print("Clearing current mask...")
        # Close any open modal first
        try:
            modal = await page.query_selector('#compareModal.active')
            if modal:
                await page.keyboard.press('Escape')
                await asyncio.sleep(0.5)
        except:
            pass

        await page.click('button:has-text("Clear Mask")')
        await asyncio.sleep(0.5)

        # Ensure PyTorch engine is selected
        print("Ensuring PyTorch engine is selected...")
        pytorch_btn = await page.query_selector('#enginePytorch')
        pytorch_class = await pytorch_btn.get_attribute('class')
        if 'active' not in pytorch_class:
            await page.click('#enginePytorch')
            await asyncio.sleep(0.5)

        # Upload the mask file
        print("Uploading mask file...")
        # Check if element exists
        mask_upload_count = await page.locator('#maskUpload').count()
        if mask_upload_count == 0:
            print("ERROR: maskUpload element not found in page")
            await browser.close()
            return False

        file_input = await page.query_selector('#maskUpload')
        await file_input.set_input_files(mask_path)
        await asyncio.sleep(1)

        # Run inpainting with uploaded mask
        print("Running inpainting with uploaded mask (PyTorch)...")
        async with page.expect_response('**/inpaint') as response_info:
            await page.click('#inpaintBtn')

        response = await response_info.value
        print(f"Inpaint response status: {response.status}")

        # Verify result
        try:
            await page.wait_for_selector('#downloadBtn', state='visible', timeout=60000)
            print("✓ SUCCESS: Inpainting completed with uploaded mask (PyTorch)")
            await browser.close()
            return True
        except:
            print("✗ FAIL: Inpainting failed with uploaded mask (PyTorch)")
            await browser.close()
            return False


async def test_mask_upload_coreml():
    """Test mask upload with CoreML engine"""

    print("\nStarting mask upload test with CoreML engine...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()

        # Navigate to the app
        print(f"Navigating to {BASE_URL}...")
        await page.goto(BASE_URL)
        await page.wait_for_load_state('networkidle')

        # Wait for page to load
        await page.wait_for_selector('#mainCanvas', timeout=5000)

        # Select a test image
        print("Selecting test image...")
        await page.select_option('#testImageSelect', '512')
        await asyncio.sleep(1)

        # Draw a mask and download it first
        print("Drawing initial mask...")
        canvas = await page.query_selector('#mainCanvas')
        box = await canvas.bounding_box()

        if box:
            await page.mouse.move(box['x'] + 100, box['y'] + 100)
            await page.mouse.down()
            await page.mouse.move(box['x'] + 200, box['y'] + 200)
            await page.mouse.up()
            await asyncio.sleep(0.5)

        # Run inpainting to generate mask for download
        print("Running inpainting to download mask...")
        async with page.expect_response('**/inpaint') as response_info:
            await page.click('#inpaintBtn')

        response = await response_info.value
        print(f"Inpaint response status: {response.status}")

        # Wait for download button and download mask
        await page.wait_for_selector('#downloadBtn', state='visible', timeout=60000)

        # Close modal if present
        try:
            modal = await page.query_selector('#compareModal.active')
            if modal:
                await page.keyboard.press('Escape')
                await asyncio.sleep(0.5)
        except:
            pass

        # Set up download handler
        mask_path = None

        async def handle_download(download):
            nonlocal mask_path
            filename = download.suggested_filename
            print(f"Downloaded: {filename}")
            if 'mask' in filename.lower():
                mask_path = await download.path()

        page.on('download', handle_download)

        # Click download
        await page.click('#downloadBtn')
        await asyncio.sleep(2)

        if not mask_path:
            print("FAIL: Could not download mask file")
            await browser.close()
            return False

        print(f"Mask saved to: {mask_path}")

        # Clear the current mask
        print("Clearing current mask...")
        # Close any open modal first
        try:
            modal = await page.query_selector('#compareModal.active')
            if modal:
                await page.keyboard.press('Escape')
                await asyncio.sleep(0.5)
        except:
            pass

        await page.click('button:has-text("Clear Mask")')
        await asyncio.sleep(0.5)

        # Switch to CoreML engine
        print("Switching to CoreML engine...")
        await page.click('#engineCoreml')
        await asyncio.sleep(0.5)

        # Upload the mask file
        print("Uploading mask file...")
        # Check if element exists
        mask_upload_count = await page.locator('#maskUpload').count()
        if mask_upload_count == 0:
            print("ERROR: maskUpload element not found in page")
            await browser.close()
            return False

        file_input = await page.query_selector('#maskUpload')
        await file_input.set_input_files(mask_path)
        await asyncio.sleep(1)

        # Run inpainting with uploaded mask
        print("Running inpainting with uploaded mask (CoreML)...")
        async with page.expect_response('**/inpaint') as response_info:
            await page.click('#inpaintBtn')

        response = await response_info.value
        print(f"Inpaint response status: {response.status}")

        # Verify result
        try:
            await page.wait_for_selector('#downloadBtn', state='visible', timeout=60000)
            print("✓ SUCCESS: Inpainting completed with uploaded mask (CoreML)")
            await browser.close()
            return True
        except:
            print("✗ FAIL: Inpainting failed with uploaded mask (CoreML)")
            await browser.close()
            return False


async def test_mask_upload_workflow():
    """Test complete workflow: draw mask -> download mask -> upload mask -> inpaint"""

    print("\nStarting complete mask upload workflow test...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()

        # Navigate to the app
        print(f"Navigating to {BASE_URL}...")
        await page.goto(BASE_URL)
        await page.wait_for_load_state('networkidle')

        # Wait for page to load
        await page.wait_for_selector('#mainCanvas', timeout=5000)

        # Select a test image
        print("1. Selecting test image...")
        await page.select_option('#testImageSelect', '512')
        await asyncio.sleep(1)

        # Draw a mask
        print("2. Drawing mask on canvas...")
        canvas = await page.query_selector('#mainCanvas')
        box = await canvas.bounding_box()

        if box:
            # Draw a rectangle mask
            await page.mouse.move(box['x'] + 150, box['y'] + 100)
            await page.mouse.down()
            await page.mouse.move(box['x'] + 250, box['y'] + 200)
            await page.mouse.up()
            await asyncio.sleep(0.5)

        # Run inpainting
        print("3. Running inpainting...")
        async with page.expect_response('**/inpaint') as response_info:
            await page.click('#inpaintBtn')

        response = await response_info.value
        print(f"   Response status: {response.status}")

        # Wait for download button
        await page.wait_for_selector('#downloadBtn', state='visible', timeout=60000)

        # Close modal
        try:
            modal = await page.query_selector('#compareModal.active')
            if modal:
                await page.keyboard.press('Escape')
                await asyncio.sleep(0.5)
        except:
            pass

        # Download both files
        print("4. Downloading inpainted image and mask...")
        download_files = []

        async def handle_download(download):
            filename = download.suggested_filename
            path = await download.path()
            download_files.append({'filename': filename, 'path': path})
            print(f"   Downloaded: {filename}")

        page.on('download', handle_download)
        await page.click('#downloadBtn')
        await asyncio.sleep(2)

        mask_path = None
        for f in download_files:
            if 'mask' in f['filename'].lower():
                mask_path = f['path']

        if not mask_path:
            print("   ✗ FAIL: Mask not downloaded")
            await browser.close()
            return False

        print(f"   Mask saved: {mask_path}")

        # Clear current result and mask
        print("5. Clearing current result...")
        # Close any open modal
        try:
            modal = await page.query_selector('#compareModal.active')
            if modal:
                await page.keyboard.press('Escape')
                await asyncio.sleep(0.5)
        except:
            pass

        # Load the image again to reset
        await page.select_option('#testImageSelect', '512')
        await asyncio.sleep(1)

        # Upload the mask
        print("6. Uploading previously saved mask...")
        # First check if element exists
        mask_upload_exists = await page.locator('#maskUpload').count() > 0
        if not mask_upload_exists:
            print("   ✗ ERROR: maskUpload element not found in page")
            await browser.close()
            return False

        file_input = await page.query_selector('#maskUpload')
        if not file_input:
            print("   ✗ ERROR: maskUpload input not found")
            await browser.close()
            return False
        await file_input.set_input_files(mask_path)
        await asyncio.sleep(1)

        # Verify mask is visible on canvas
        print("7. Verifying mask is applied...")
        # Check if coverage info shows mask applied
        coverage_elem = await page.query_selector('#infoCoverage')
        coverage_text = await coverage_elem.inner_text()
        print(f"   Coverage: {coverage_text}")

        if '%' in coverage_text and coverage_text != '0%':
            print("   ✓ Mask appears to be loaded")
        else:
            print("   ✗ Mask may not have loaded correctly")
            await browser.close()
            return False

        # Run inpainting with uploaded mask
        print("8. Running inpainting with uploaded mask...")
        async with page.expect_response('**/inpaint') as response_info:
            await page.click('#inpaintBtn')

        response = await response_info.value
        print(f"   Response status: {response.status}")

        # Verify result
        try:
            await page.wait_for_selector('#downloadBtn', state='visible', timeout=60000)
            print("   ✓ Inpainting completed successfully!")
            await browser.close()
            return True
        except:
            print("   ✗ Inpainting failed")
            await browser.close()
            return False


async def main():
    """Run all tests"""
    print("=" * 60)
    print("Mask Upload Functionality Test")
    print("=" * 60)
    print()

    # Check if server is running
    import requests
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Server is running: {response.json()}")
        print()
    except Exception as e:
        print(f"ERROR: Cannot connect to server at {BASE_URL}")
        print(f"Please start the server first: python launch_gpu_service_mac.sh")
        print(f"Error: {e}")
        return

    # Run tests
    result1 = await test_mask_upload_pytorch()
    result2 = await test_mask_upload_coreml()
    result3 = await test_mask_upload_workflow()

    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print(f"  Mask upload (PyTorch): {'PASS' if result1 else 'FAIL'}")
    print(f"  Mask upload (CoreML): {'PASS' if result2 else 'FAIL'}")
    print(f"  Complete workflow: {'PASS' if result3 else 'FAIL'}")
    print("=" * 60)


if __name__ == '__main__':
    asyncio.run(main())

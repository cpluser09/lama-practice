#!/usr/bin/env python3
"""
Test download functionality using Playwright
Tests that both inpainted image and mask are downloaded
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

from playwright.async_api import async_playwright

BASE_URL = "http://localhost:5002"


async def test_download_with_mask():
    """Test that download includes both inpainted image and mask"""

    print("Starting Playwright test for download functionality...")

    async with async_playwright() as p:
        # Launch browser with download tracking
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()

        # Navigate to the app
        print(f"Navigating to {BASE_URL}...")
        await page.goto(BASE_URL)

        # Wait for page to load
        await page.wait_for_load_state('networkidle')

        # Check if server is running
        try:
            await page.wait_for_selector('#mainCanvas', timeout=5000)
        except Exception as e:
            print(f"Error: Page did not load properly. Is the server running?")
            print(f"Error: {e}")
            await browser.close()
            return False

        # Select a test image from dropdown
        print("Selecting test image...")
        await page.select_option('#testImageSelect', '512')

        # Wait for image to load
        await asyncio.sleep(1)

        # Draw a mask on the canvas
        print("Drawing mask on canvas...")
        canvas = await page.query_selector('#mainCanvas')
        box = await canvas.bounding_box()

        if box:
            # Draw a simple rectangle mask
            await page.mouse.move(box['x'] + 100, box['y'] + 100)
            await page.mouse.down()
            await page.mouse.move(box['x'] + 200, box['y'] + 200)
            await page.mouse.up()

            await asyncio.sleep(0.5)

        # Click the Inpaint button
        print("Starting inpainting...")
        async with page.expect_response('**/inpaint') as response_info:
            await page.click('#inpaintBtn')

        response = await response_info.value
        print(f"Inpaint response status: {response.status}")

        # Wait for the download button to appear
        print("Waiting for download button...")
        try:
            await page.wait_for_selector('#downloadBtn', state='visible', timeout=60000)
            print("Download button appeared!")
        except Exception as e:
            print(f"Error: Download button did not appear in time")
            print(f"Error: {e}")
            await browser.close()
            return False

        # Close the compare modal if it's blocking the download button
        try:
            modal = await page.query_selector('#compareModal.active')
            if modal:
                print("Closing compare modal...")
                # Press ESC to close the modal
                await page.keyboard.press('Escape')
                await asyncio.sleep(0.5)
        except:
            pass

        # Set up download handler before clicking
        download_files = []

        async def handle_download(download):
            filename = download.suggested_filename
            print(f"Download started: {filename}")
            path = await download.path()
            download_files.append({
                'filename': filename,
                'path': path
            })
            print(f"Download saved to: {path}")

        page.on('download', handle_download)

        # Click the download button
        print("Clicking download button...")
        await page.click('#downloadBtn')

        # Wait for downloads to complete
        await asyncio.sleep(2)

        # Verify downloads
        print("\n=== Download Verification ===")
        print(f"Number of files downloaded: {len(download_files)}")

        if len(download_files) == 0:
            print("FAIL: No files were downloaded")
            await browser.close()
            return False

        for f in download_files:
            print(f"  - {f['filename']}")

        # Check for expected files
        has_inpaint = any('inpaint' in f['filename'].lower() for f in download_files)
        has_mask = any('mask' in f['filename'].lower() for f in download_files)

        print(f"\nInpainted image downloaded: {has_inpaint}")
        print(f"Mask downloaded: {has_mask}")

        if has_inpaint and has_mask:
            print("\n✓ SUCCESS: Both inpainted image and mask were downloaded!")
            await browser.close()
            return True
        elif has_inpaint:
            print("\n✗ PARTIAL: Only inpainted image was downloaded (mask missing)")
            await browser.close()
            return False
        else:
            print("\n✗ FAIL: Expected files were not downloaded")
            await browser.close()
            return False


async def test_download_without_mask():
    """Test download without drawing a mask (should only download inpainted image)"""

    print("\nStarting test for download without mask...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()

        await page.goto(BASE_URL)
        await page.wait_for_load_state('networkidle')

        # Select test image
        await page.select_option('#testImageSelect', '512')
        await asyncio.sleep(1)

        # Click inpaint without drawing mask
        print("Starting inpainting without mask...")
        async with page.expect_response('**/inpaint') as response_info:
            await page.click('#inpaintBtn')

        response = await response_info.value
        print(f"Inpaint response status: {response.status}")

        # Wait for download button
        await page.wait_for_selector('#downloadBtn', state='visible', timeout=60000)

        # Close the compare modal if it's blocking the download button
        try:
            modal = await page.query_selector('#compareModal.active')
            if modal:
                print("Closing compare modal...")
                await page.keyboard.press('Escape')
                await asyncio.sleep(0.5)
        except:
            pass

        # Set up download handler
        download_files = []

        async def handle_download(download):
            filename = download.suggested_filename
            print(f"Download started: {filename}")
            path = await download.path()
            download_files.append({'filename': filename, 'path': path})

        page.on('download', handle_download)

        # Click download
        await page.click('#downloadBtn')
        await asyncio.sleep(2)

        print(f"\nFiles downloaded: {len(download_files)}")
        for f in download_files:
            print(f"  - {f['filename']}")

        # When no mask was drawn, only inpainted image should be downloaded
        has_inpaint = any('inpaint' in f['filename'].lower() for f in download_files)
        has_mask = any('mask' in f['filename'].lower() for f in download_files)

        print(f"\nInpainted image downloaded: {has_inpaint}")
        print(f"Mask downloaded: {has_mask}")

        if has_inpaint and not has_mask:
            print("\n✓ SUCCESS: Only inpainted image was downloaded (as expected)")
            await browser.close()
            return True
        else:
            print("\n✗ FAIL: Unexpected download behavior")
            await browser.close()
            return False


async def main():
    """Run all tests"""
    print("=" * 60)
    print("LaMa Download Functionality Test")
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
        print(f"Please start the server first: python launch-cpu-service_docker.py")
        print(f"Error: {e}")
        return

    # Run tests
    result1 = await test_download_with_mask()
    result2 = await test_download_without_mask()

    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print(f"  Download with mask: {'PASS' if result1 else 'FAIL'}")
    print(f"  Download without mask: {'PASS' if result2 else 'FAIL'}")
    print("=" * 60)


if __name__ == '__main__':
    asyncio.run(main())

package com.example.nehadam;

import com.example.nehadam.controller.ImageController;
import com.example.nehadam.service.ImageService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.servlet.MockMvc;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest
class ImageFileControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Mock
    private ImageService imageService;

    @InjectMocks
    private ImageController imageFileController;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testUploadImage() throws Exception {
        MockMultipartFile mockFile = new MockMultipartFile(
                "file",
                "test-image.jpg",
                MediaType.IMAGE_JPEG_VALUE,
                "Test image content".getBytes()
        );

        String email = "test@example.com";

        // Mock the service behavior if necessary
        // when(imageService.saveFile(any(), anyString())).thenReturn(...);

        mockMvc.perform(multipart("/images/upload")
                        .file(mockFile)
                        .param("email", email))
                .andExpect(status().isOk());

        // Additional assertions can be made based on the expected behavior
    }
}

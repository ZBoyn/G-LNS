// G-LNS Website Scripts

// Gallery/Carousel Data
const slides = [
    {
        title: "Motivation",
        description: "Limitations of existing AHD approaches and the proposed G-LNS framework concept.",
        description2: "",
        image: "assets/images/introduction.png" 
    },
    {
        title: "Method Overview", 
        description: "The cyclic process of G-LNS: Initialization, Evaluation, Population Management, and Evolution.",
        description2: "",
        image: "assets/images/method.png"
    },
    {
        title: "Convergence and Evolutionary Analysis",
        description: "(a) Evolutionary Progress: Validation score trajectory of the best operator over 200 generations; the steady decline confirms the LLM's capacity to evolve high-performance heuristics.",
        description2: "(b) Evaluation Progress: Convergence comparison on CVRP100 instances. GLNS identifies the best solution in 70s across all 64 instances, significantly outperforming both the Solver (320s) and MCTSAHD(ACO) (1110s) in terms of search efficiency.",
        image: "assets/images/convergence.png"
    },
    {
        title: "Case Study",
        description: "A snapshot of the best operator discovered by G-LNS on CVRP50 instances.",
        description2: "",
        image: "assets/images/case_study.png"
    },
    {
        title: "Performance comparison on TSP and CVRP instances",
        description: "GLNS outperforms LLM-based AHD methods and Solver in terms of solution quality and search efficiency on TSP and CVRP instances.",
        description2: "",
        image: "assets/images/exp.png"
    },
    {
        title: "Performance comparison on Benchmark Instances",
        description: "GLNS achieves the best performance on all benchmark instances, significantly outperforming other LLM-based AHD methods.",
        description2: "",
        image: "assets/images/benchmarks.png"
    }
];

let currentSlideIndex = 0;

function initializeGallery() {
    const track = document.getElementById('galleryTrack');
    if (!track) return; // Exit if no gallery present

    // clear existing
    track.innerHTML = '';

    slides.forEach((slide, index) => {
        const slideDiv = document.createElement('div');
        slideDiv.className = 'gallery-slide';
        if (index === 0) slideDiv.classList.add('active');
        
        slideDiv.innerHTML = `
            <div class="slide-content">
                <img src="${slide.image}" alt="${slide.title}" class="gallery-image" onclick="openModal(this)">
                <div class="gallery-caption">${slide.title}</div>
                <div class="gallery-desc">${slide.description}</div>
                <div class="gallery-desc">${slide.description2}</div>
            </div>
        `;
        track.appendChild(slideDiv);
    });

    updateSlideControls();
}

function updateSlideControls() {
    const track = document.getElementById('galleryTrack');
    if (track) {
        track.style.transform = `translateX(-${currentSlideIndex * 100}%)`;
    }
}

function nextSlide() {
    if (currentSlideIndex < slides.length - 1) {
        currentSlideIndex++;
    } else {
        // Loop back to start
        currentSlideIndex = 0;
    }
    updateSlideControls();
}

function previousSlide() {
    if (currentSlideIndex > 0) {
        currentSlideIndex--;
    } else {
        // Loop to end
        currentSlideIndex = slides.length - 1;
    }
    updateSlideControls();
}

// Image Modal Functions
function openModal(img) {
    const modal = document.getElementById('imageModal');
    const modalImage = document.getElementById('modalImage');
    
    if (modal && modalImage) {
        modal.style.display = 'flex';
        modalImage.src = img.src;
        
        document.body.style.overflow = 'hidden';
        
        // Close when clicking background
        modal.onclick = function(e) {
            if (e.target === modal || e.target.classList.contains('modal-close')) {
                closeModal();
            }
        };
    }
}

function closeModal() {
    const modal = document.getElementById('imageModal');
    if (modal) {
        modal.style.display = 'none';
        document.body.style.overflow = 'auto';
    }
}

// Copy Code Functionality
function copyCode(button) {
    const codeBlock = button.closest('.code-block');
    const code = codeBlock.querySelector('pre code') || codeBlock.querySelector('pre');
    const text = code.innerText;
    
    navigator.clipboard.writeText(text).then(() => {
        const originalIcon = button.innerHTML;
        button.innerHTML = '<i class="fas fa-check"></i>';
        button.style.color = '#10b981';
        
        setTimeout(() => {
            button.innerHTML = originalIcon;
            button.style.color = '';
        }, 2000);
    });
}

document.addEventListener('DOMContentLoaded', function() {
    initializeGallery();
    
    // Key listeners
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') closeModal();
        if (e.key === 'ArrowLeft') previousSlide();
        if (e.key === 'ArrowRight') nextSlide();
    });

    // Mobile Nav Toggle
    const navToggle = document.querySelector('.nav-toggle');
    const navMenu = document.querySelector('.nav-menu');
    
    if (navToggle && navMenu) {
        navToggle.addEventListener('click', function() {
            navMenu.classList.toggle('active'); // You might need CSS for .active on mobile
            navToggle.classList.toggle('active');
        });
    }

    // Smooth Scroll
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                // Close mobile menu if open
                if (navMenu && navMenu.classList.contains('active')) {
                    navMenu.classList.remove('active');
                    if(navToggle) navToggle.classList.remove('active');
                }

                const headerOffset = 80;
                const elementPosition = target.getBoundingClientRect().top;
                const offsetPosition = elementPosition + window.pageYOffset - headerOffset;
                
                window.scrollTo({
                    top: offsetPosition,
                    behavior: "smooth"
                });
            }
        });
    });
});

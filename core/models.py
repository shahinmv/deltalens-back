from django.db import models
from django.contrib.auth.models import AbstractUser

class MarketStats(models.Model):
    market_cap = models.FloatField()
    market_cap_change_24h = models.FloatField()
    market_cap_change_percentage_24h = models.FloatField()
    btc_dominance = models.FloatField()
    btc_dominance_change_24h = models.FloatField()
    volume_24h = models.FloatField()
    volume_24h_change = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name_plural = "Market Stats"

    def __str__(self):
        return f"Market Stats - {self.updated_at}"

class User(AbstractUser):
    ROLE_CHOICES = [
        ('non_member', 'Non-Member'),
        ('member', 'Member'),
        ('admin', 'Admin'),
    ]
    
    role = models.CharField(
        max_length=20,
        choices=ROLE_CHOICES,
        default='non_member'
    )
    email = models.EmailField(unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']
    
    def __str__(self):
        return self.email
    
    def can_access_dashboard(self):
        return self.role in ['member', 'admin'] or self.is_superuser
    
    def is_admin_user(self):
        return self.role == 'admin' or self.is_superuser
    
    def save(self, *args, **kwargs):
        # Automatically assign admin role to superusers
        if self.is_superuser and self.role == 'non_member':
            self.role = 'admin'
        super().save(*args, **kwargs)
